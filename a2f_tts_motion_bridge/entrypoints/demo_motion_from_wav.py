import argparse

from data.a2f.a2f_tts_motion_bridge.motion_core.audio_stream import MockStreamConfig, iter_mock_tts_stream_from_wav
from data.a2f.a2f_tts_motion_bridge.motion_core.types import SessionConfig
from data.a2f.a2f_tts_motion_bridge.motion_core.motion_session import MotorStreamSession


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_wav", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out_npy", required=True)
    parser.add_argument("--emotion", default="neutral")
    parser.add_argument("--intensity", type=float, default=1.0)
    parser.add_argument("--chunk_sec_min", type=float, default=0.10)
    parser.add_argument("--chunk_sec_max", type=float, default=0.30)
    parser.add_argument("--force_cpu", action="store_true")
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()

    cfg = SessionConfig(
        session_id="demo-session",
        emotion=args.emotion,
        intensity=args.intensity,
        output_fps=30.0,
        save_npy=True,
        npy_save_dir=".",
    )
    session = MotorStreamSession(args.model, cfg, force_cpu=args.force_cpu, warmup=args.warmup)
    stream_cfg = MockStreamConfig(
        emotion=args.emotion,
        intensity=args.intensity,
        chunk_sec_min=args.chunk_sec_min,
        chunk_sec_max=args.chunk_sec_max,
    )
    total_batches = 0
    total_frames = 0
    for chunk in iter_mock_tts_stream_from_wav(args.input_wav, session_id=cfg.session_id, config=stream_cfg):
        frames = session.process_chunk(chunk)
        if frames:
            total_batches += 1
            total_frames += len(frames)
    tail_frames = session.finalize()
    total_frames += len(tail_frames)
    session.recorder.save(args.out_npy)
    print(f"[OK] saved: {args.out_npy}")
    print(f"[INFO] batches={total_batches}, frames={total_frames}")
    print(f"[INFO] infer_frames_native={session.engine.infer_frames}")
    print(f"[INFO] provider={session.engine.providers}")


if __name__ == "__main__":
    main()
