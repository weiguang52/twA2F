"""Independent prompt -> motor NPY service using existing VoiceAgent wire types."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import threading
import time
import grpc
from ..motion_core.standalone_face import PromptFacePlanner, PromptFaceSession
from ..protocol.voiceagent_generated import ActionImageMediaStream_pb2 as pb
from ..protocol.voiceagent_generated import ActionImageMediaStream_pb2_grpc as rpc


class PromptFaceService(rpc.VoiceAgentServicer):
    def __init__(self, planner=None):
        self.planner = planner
        self.lock = threading.Lock()

    def StreamSession(self, request_iterator, context):
        request_id = ''
        try:
            req = next(iter(request_iterator))
            request_id = req.request_id or 'prompt-face'
            if req.audio_chunk or not req.end_of_stream:
                raise ValueError('Send one audio-free request with end_of_stream=true')
            meta = dict(req.meta)
            if set(meta)-{'prompt', 'face_plan', 'export_fps', 'segment_ms'}:
                raise ValueError('Unknown prompt-face meta field')
            if ('prompt' in meta) == ('face_plan' in meta):
                raise ValueError('Provide exactly one of prompt or face_plan')
            if 'prompt' in meta:
                if self.planner is None:
                    raise ValueError('Prompt model not configured; only validated plans accepted')
                with self.lock:
                    plan, _ = self.planner.parse(meta['prompt'])
            else:
                plan = meta['face_plan']
            session = PromptFaceSession(plan, request_id)
            for segment in session.segments(int(meta.get('export_fps', 30)), int(meta.get('segment_ms', 640))):
                if not context.is_active():
                    return
                yield pb.AudioResponse(request_id=request_id, ts=int(time.time()*1000),
                    type=pb.ACTION, action_npz=segment.npy_bytes,
                    action_description=f'Prompt face segment #{segment.segment_index}',
                    model_version=f'motorstream|segment_ms={segment.end_ms-segment.start_ms}|segment_idx={segment.segment_index}|start_ms={segment.start_ms}|end_ms={segment.end_ms}|final={int(segment.is_final)}',
                    end_of_response=False)
            yield pb.AudioResponse(request_id=request_id, ts=int(time.time()*1000), end_of_response=True)
        except StopIteration:
            return
        except (ValueError, TypeError) as exc:
            yield pb.AudioResponse(request_id=request_id, ts=int(time.time()*1000), type=pb.ERROR,
                                   error_msg=str(exc), end_of_response=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', help='Local Qwen3 model; omitted means plan-only mode')
    p.add_argument('--device', default='cuda:1')
    p.add_argument('--listen', default='127.0.0.1:50062')
    args = p.parse_args()
    planner = PromptFacePlanner(args.base, args.device) if args.base else None
    server = grpc.server(ThreadPoolExecutor(max_workers=2))
    rpc.add_VoiceAgentServicer_to_server(PromptFaceService(planner), server)
    if not server.add_insecure_port(args.listen):
        raise RuntimeError('Cannot bind prompt-face service')
    server.start()
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        server.stop(2).wait()


if __name__ == '__main__':
    main()
