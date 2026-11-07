"""ALFWorld-final GLM hint protocol, with only native shopping state/actions adapted."""
import http.client
import json
import os
from pathlib import Path
import re
import socket
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from common import ROOT, atomic_json
from . import rethink_api_protocol as protocol
from .rethink_benchmark_contract import command_allowed

MODEL=protocol.MODEL
_SESSION_ID='searchqa-glm-hints-'+uuid.uuid4().hex


class HintAPIHold(RuntimeError):
    """Retain the same rollout/pre-update Actor; never silently discard hints."""
    def __init__(self,reason,temporary=False,diagnostics=None):
        self.reason=reason
        self.temporary=temporary
        self.diagnostics=diagnostics or {}
        self.fatal=False
        super().__init__(reason)


class HintAPIFatal(HintAPIHold):
    """Actionable local/configuration failure: never update or retry the API."""
    def __init__(self,reason,diagnostics=None):
        super().__init__(reason,diagnostics=diagnostics)
        self.fatal=True


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):
        raise HintAPIFatal('api_redirect_refused',{'stage':'open'})


def trust_environment_proxy():
    value=os.environ.get('RETHINK_HINT_TRUST_ENV','true').strip().lower()
    if value not in ('true','false','1','0'):
        raise HintAPIFatal('api_invalid_proxy_option',{'stage':'configuration'})
    return value in ('true','1')


def transport_opener():
    handlers=[NoRedirect()]
    if not trust_environment_proxy():
        handlers.insert(0,urllib.request.ProxyHandler({}))
    return urllib.request.build_opener(*handlers).open


def error_details(exc,stage):
    """Fixed safe fields only: never stringify exceptions, headers or bodies."""
    cause=exc.reason if isinstance(exc,urllib.error.URLError) else exc
    kind='network_error'
    if isinstance(cause,socket.gaierror):kind='dns_error'
    elif isinstance(cause,ssl.SSLCertVerificationError):kind='tls_certificate_error'
    elif isinstance(cause,ssl.SSLError):kind='tls_error'
    elif isinstance(cause,TimeoutError):kind='timeout'
    elif isinstance(cause,http.client.RemoteDisconnected):kind='remote_disconnected'
    elif isinstance(cause,http.client.IncompleteRead):kind='incomplete_response'
    elif isinstance(cause,ConnectionRefusedError):kind='connection_refused'
    elif isinstance(cause,ConnectionResetError):kind='connection_reset'
    details={'stage':stage,'error_kind':kind}
    number=getattr(cause,'errno',None)
    if isinstance(number,int):details['errno']=number
    return details


def event(name,**details):
    print('[RETHINK API] '+json.dumps({'event':name,**details},sort_keys=True),flush=True)


def commands(available):
    if isinstance(available,list):return list(available)
    result=['search[<your query>]'] if available.get('has_search_bar') else []
    result.extend('click['+str(name)+']' for name in available.get('clickables',[]))
    return result


def allowed(action,available):
    return command_allowed(action,commands(available),'webshop')


def prepare_group(episode):
    if episode.get('benchmark') != 'search':
        raise ValueError('Expected a native Search-QA trajectory')
    steps=[]
    for step in episode['steps']:
        steps.append({'step':int(step['step']),
                      'observation':step['observation'],'admissible_actions':[],
                      'reasoning':step['reasoning'],'action':step['action'],
                      'feedback':step['feedback']})
    canonical={'benchmark':'search','task':episode['task'],
               'steps':steps}
    targets={str(int(s['step'])):[str(int(s['step']))] for s in episode['steps'] if s.get('triggered')}
    if len({s['step'] for s in steps})!=len(steps):
        raise ValueError('Duplicate captured step ID')
    return {'key':protocol.digest(canonical),'episode':canonical,'targets':targets}


def extract_content(data):
    choice=data['choices'][0]
    content=choice['message'].get('content')
    if isinstance(content,list):
        content=''.join(x.get('text','') for x in content if isinstance(x,dict) and x.get('type') in ('text','output_text'))
    if not isinstance(content,str) or not content.strip():raise ValueError('empty_content')
    return content


def parsed_hints(content,group,finish_reason):
    outputs=protocol.parse_hints(content,group,finish_reason)
    valid={};failures={}
    for key,record in outputs.items():
        if record['payload'] is not None:
            valid[int(key)]='Reasoning skill: '+record['payload']['reasoning_skill'].strip()
        else:
            failures[int(key)]=record['error']
    return valid,failures


def validate_hints(raw,episode):
    return parsed_hints(json.dumps(raw,ensure_ascii=False),prepare_group(episode),'stop')


def validate_auth(cfg):
    if not isinstance(cfg,dict) or not isinstance(cfg.get('base'),str):
        raise ValueError('Hint API configuration must contain an HTTPS base URL')
    parts=urllib.parse.urlsplit(cfg['base'])
    if parts.scheme!='https' or not parts.netloc or parts.username or parts.password or parts.query or parts.fragment:
        raise ValueError('Use a credential-free HTTPS API base URL')
    if any(c.isspace() for c in cfg['base']) or parts.path.rstrip('/').endswith('/chat/completions'):
        raise ValueError('API base must not contain whitespace or the chat/completions suffix')
    parts.port  # Reject malformed ports without a request.
    if not isinstance(cfg.get('key'),str) or not cfg['key'].strip() or any(c in cfg['key'] for c in '\r\n') or cfg.get('model')!=MODEL:
        raise ValueError('ALFWorld-final recipe requires the explicit glm-5.3-flash model')
    return cfg


def auth():
    path=Path(os.environ['RETHINK_HINT_AUTH'])
    if os.name!='nt' and path.stat().st_mode & 0o077:
        raise PermissionError('Hint credential file must be owner-only')
    return validate_auth(json.loads(path.read_text(encoding='utf-8')))


def fetch(episode,cache_dir,timeout=120,retries=5,opener=None,sleeper=time.sleep):
    """Same final model/body/parser and six-attempt transport boundary as ALFWorld.

    Invalid returned hints use the ordinary static-skill Teacher. Transport,
    transient failures HOLD this batch instead of dropping local hints. Local
    I/O, certificate and permanent HTTP errors stop visibly without an update.
    """
    try:cfg=auth()
    except (KeyError,OSError,ValueError,TypeError):
        raise HintAPIFatal('api_not_configured',{'stage':'configuration'}) from None
    if timeout<=0 or retries<0:
        raise HintAPIFatal('api_invalid_retry_options',{'stage':'configuration'})
    group=prepare_group(episode)
    if not group['targets']:return {},{}
    body=protocol.request_body(group)
    url=cfg['base'].rstrip('/')+'/chat/completions'
    identity=protocol.digest({'body':body,'protocol':protocol.VERSION,'endpoint':url})
    cache=Path(cache_dir)/(identity+'.json')
    if cache.exists():
        try:saved=json.loads(cache.read_text(encoding='utf-8'))
        except (OSError,ValueError):
            raise HintAPIFatal('api_cache_read_error',{'stage':'cache_read','request_sha256':identity}) from None
        if not isinstance(saved,dict) or saved.get('request_sha256')!=identity:
            raise HintAPIFatal('api_cache_identity_mismatch',{'stage':'cache_read','request_sha256':identity})
        return parsed_hints(saved.get('content'),group,saved.get('finish_reason'))
    headers={'Content-Type':'application/json','Accept':'application/json',
             'User-Agent':'AlfworldHintEvaluation/1.0','Authorization':'Bearer '+cfg['key']}
    if urllib.parse.urlsplit(cfg['base']).hostname=='opencode.ai' and '/inference/go/' in url:
        headers['x-opencode-session']=_SESSION_ID
    opener=opener or transport_opener()
    reason='api_transport_error'
    details={}
    started=time.perf_counter()
    for attempt in range(retries+1):
        stage='open'
        tick=time.perf_counter()
        try:
            req=urllib.request.Request(url,json.dumps(body,ensure_ascii=False).encode(),headers)
            with opener(req,timeout=timeout) as response:
                stage='read'
                raw=response.read(4*1024**2+1)
            if len(raw)>4*1024**2:
                raise HintAPIFatal('api_response_too_large',{'stage':'read','request_sha256':identity})
            stage='decode'
            data=json.loads(raw)
            if not isinstance(data,dict) or not isinstance(data.get('choices'),list) or not data['choices']:
                raise HintAPIFatal('api_malformed_response',{'stage':'decode','request_sha256':identity})
        except urllib.error.HTTPError as exc:
            reason='http_'+str(exc.code)
            details={'stage':'open','http_status':exc.code,'request_sha256':identity}
            exc.close()
            if exc.code not in (408,429,500,502,503,504):
                raise HintAPIFatal(reason,details) from None
        except (OSError,urllib.error.URLError,ConnectionError,http.client.RemoteDisconnected,
                http.client.HTTPException) as exc:
            reason='api_transport_error'
            details={**error_details(exc,stage),'request_sha256':identity}
            if details['error_kind']=='tls_certificate_error':
                raise HintAPIFatal('api_tls_certificate_error',details) from None
        except (json.JSONDecodeError,UnicodeDecodeError):
            reason='api_invalid_json'
            details={'stage':'decode','request_sha256':identity}
        else:
            # Keep local disk operations OUTSIDE the network retry handler.
            try:
                content=extract_content(data)
                finish=data['choices'][0].get('finish_reason')
            except (KeyError,IndexError,TypeError,ValueError,AttributeError):
                content=None
                finish=data['choices'][0].get('finish_reason') if isinstance(data['choices'][0],dict) else None
            result=parsed_hints(content,group,finish)
            try:
                atomic_json(cache,{'request_sha256':identity,'schema':protocol.VERSION,'model':MODEL,
                                  'content':content,'finish_reason':finish,'usage':data.get('usage',{})})
            except OSError:
                raise HintAPIFatal('api_cache_write_error',{'stage':'cache_write','request_sha256':identity,
                                  'api_response_received':True,'actor_updated':False}) from None
            event('request_complete',request_sha256=identity,attempt=attempt+1,
                  seconds=round(time.perf_counter()-started,3),accepted_hints=len(result[0]))
            return result
        details.update(attempt=attempt+1,max_attempts=retries+1,
                       attempt_seconds=round(time.perf_counter()-tick,3),
                       elapsed_seconds=round(time.perf_counter()-started,3))
        delay=min(60,2**(attempt+1)) if attempt<retries else 0
        event('request_retry' if delay else 'request_failed',reason=reason,retry_after_s=delay,**details)
        if delay:sleeper(delay)
    raise HintAPIHold(reason,temporary=True,diagnostics=details) from None


def main(argv=None):
    """Standalone CPU/API check; no framework, model weights or GPU required."""
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true',required=True)
    parser.add_argument('--allow-api-cost',action='store_true',help='Also send one synthetic hint request')
    parser.add_argument('--cache-dir',default='runs/hint_api_check')
    args=parser.parse_args(argv)
    try:
        if os.environ.get('RETHINK_HINT_AUTH'):
            cfg=auth()
        else:
            cfg=validate_auth({'base':os.environ['HINT_API_BASE'],
                               'key':os.environ['HINT_API_KEY'],'model':os.environ['HINT_MODEL']})
        cache=(ROOT/args.cache_dir).resolve()
        if not cache.is_relative_to(ROOT/'runs'):
            raise ValueError('Use a cache directory under this checkout\'s runs')
        proxy_mode=trust_environment_proxy()
        atomic_json(cache/'check_status.json',{'status':'local_checks_passed','paid_request':False})
    except HintAPIHold as exc:
        event('check_failed',reason=exc.reason,**exc.diagnostics);return 2
    except (KeyError,OSError,ValueError,TypeError):
        event('check_failed',reason='configuration_or_cache_error',stage='local_check');return 2
    event('local_checks_passed',trust_environment_proxy=proxy_mode,paid_request=False)
    if not args.allow_api_cost:return 0
    episode={'benchmark':'search','task':'In which city is the fictional River Museum?',
             'steps':[{'step':0,'triggered':True,'observation':'No search results yet. Synthetic fixture.',
                       'reasoning':'I might be confusing the museum with another landmark.',
                       'action':'<search>River Museum location</search>',
                       'feedback':'The search returned a snippet about a different museum.'}]}
    # Use the exact training transport/parser. Only credential loading is local
    # to this CLI; never put the key into argv, resolved configs or diagnostics.
    original=globals()['auth']
    globals()['auth']=lambda:cfg
    try:
        values,failures=fetch(episode,cache/'checks'/uuid.uuid4().hex,retries=0)
        event('check_complete',accepted_hints=len(values),fallback_count=len(failures),
              actor_updated=False,synthetic_fixture=True)
        return 0 if len(values)==1 and not failures else 1
    except HintAPIHold as exc:
        event('check_failed',reason=exc.reason,**exc.diagnostics);return 2
    finally:
        globals()['auth']=original


if __name__=='__main__':
    raise SystemExit(main())
