"""Real native SearchEnv/HTTP/parser/reward smoke with a labeled CPU fixture.

This is NOT an E5/FAISS quality evaluation or an Actor benchmark.
"""
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import sys
from threading import Thread
from common import ROOT, UPSTREAM, atomic_json
from setup import verify


@contextmanager
def fixture_retriever():
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            requests.append(request)
            rows = [{'document': {'id': str(i), 'contents': '"France"\nParis is the capital of France.'},
                     'score': float(3-i)} for i in range(request['topk'])]
            body = json.dumps({'result': [rows]}).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}/retrieve', requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def smoke():
    verify()
    sys.path.insert(0, str(UPSTREAM))
    from omegaconf import OmegaConf
    from agent_system.environments.env_package.search.third_party.skyrl_gym.envs.search.env import SearchEnv
    from agent_system.environments.env_package.search import search_projection
    with fixture_retriever() as (url, requests):
        env = SearchEnv(OmegaConf.create({'search_url': url, 'topk': 3, 'timeout': 5, 'log_requests': False}))
        env.reset({'ground_truth': {'target': ['Paris']}, 'max_turns': 4, 'data_source': 'fixture'})
        action, valid = search_projection(['<think>Find evidence.</think><search>capital of France</search>'])
        step = env.step(action[0])
        assert valid == [1] and step['reward'] == 0 and not step['done']
        assert 'Paris' in step['observations'][0]['content'] and '<information>' in step['observations'][0]['content']
        action, valid = search_projection(['<think>Use the evidence.</think><answer>PARIS!</answer>'])
        final = env.step(action[0])
        assert valid == [1] and final['done'] and final['reward'] == 1.
        assert len(requests) == 1 and requests[0]['topk'] == 3
        env.reset({'ground_truth': {'target': ['Paris']}, 'max_turns': 4, 'data_source': 'fixture'})
        wrong = env.step('<answer>London</answer>')
        assert wrong['done'] and wrong['reward'] == 0
        env.close()
    result = {'status': 'passed', 'fixture': True, 'native_http_search': True,
        'retrieved_information_returned': True, 'correct_normalized_answer_reward': 1.,
        'wrong_answer_reward': 0., 'actor_model_used': False,
        'e5_faiss_tested': False, 'gpu_training_tested': False}
    atomic_json(ROOT / 'results/environment_smoke.json', result)
    return result


if __name__ == '__main__':
    print(json.dumps(smoke(), indent=2))
