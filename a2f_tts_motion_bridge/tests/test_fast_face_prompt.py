import json
import unittest
from unittest.mock import Mock
from a2f_tts_motion_bridge.motion_core.fast_face_prompt import parse_fast_face,ALIASES
from a2f_tts_motion_bridge.entrypoints.prompt_face_server import PromptFaceService,pb

class FastFaceTests(unittest.TestCase):
    def test_all_aliases(self):
        for text,(kind,side) in ALIASES.items():
            with self.subTest(text=text):
                a=parse_fast_face(text)['actions'][0]
                self.assertEqual((a['action'],a['side']),(kind,side))

    def test_modifiers_and_duration(self):
        a=parse_fast_face('请温柔地轻微微笑，保持2秒。')['actions'][0]
        self.assertEqual((a['strength'],a['tone'],a['hold_ms'],a['attack_ms']),(.25,'warm',2000,500))
        self.assertEqual(parse_fast_face('愤怒地皱眉')['actions'][0]['strength'],.6)
        a=parse_fast_face('左眼眨一下')['actions'][0]
        self.assertEqual((a['strength'],a['attack_ms'],a['hold_ms'],a['release_ms']),(1.,80,40,140))

    def test_no_partial_matching(self):
        for text in ['不要微笑','微笑然后闭眼','如果我说微笑就闭眼','抬头并抬眉',
                     '微笑但不要眯眼','重复眨眼三次','左眼眨一下右眼保持睁开',
                     '微笑，强度0.7','像刚得到礼物一样微笑','请不要皱眉','微笑\n闭眼']:
            with self.subTest(text=text):self.assertIsNone(parse_fast_face(text))
        for text in ['',None,'微笑保持11秒','微笑保持0.0001秒']:
            with self.subTest(text=text),self.assertRaises(ValueError):parse_fast_face(text)

    def test_fast_bypasses_model_and_route_is_visible(self):
        planner=Mock();ctx=Mock();ctx.is_active.return_value=True
        req=pb.AudioRequest(request_id='fast',end_of_stream=True,meta={'prompt':'左眼眨一下','export_fps':'90'})
        replies=list(PromptFaceService(planner,True).StreamSession(iter([req]),ctx))
        planner.parse.assert_not_called()
        self.assertTrue(any(r.type==pb.ACTION and 'source=fast_grammar' in r.action_description for r in replies))
        self.assertTrue(replies[-1].end_of_response)

    def test_fallback_gets_original_and_default_unchanged(self):
        text='像刚得到礼物一样微笑';planner=Mock()
        planner.parse.return_value=(json.dumps(parse_fast_face('微笑')),0.)
        ctx=Mock();ctx.is_active.return_value=True
        req=pb.AudioRequest(request_id='fallback',end_of_stream=True,meta={'prompt':text})
        replies=list(PromptFaceService(planner,True).StreamSession(iter([req]),ctx))
        planner.parse.assert_called_once_with(text)
        self.assertIn('source=qwen3',replies[0].action_description)
        req.meta['prompt']='微笑'
        replies=list(PromptFaceService().StreamSession(iter([req]),ctx))
        self.assertEqual(replies[0].type,pb.ERROR)

    def test_unmatched_without_model_errors(self):
        req=pb.AudioRequest(request_id='unknown',end_of_stream=True,meta={'prompt':'不要微笑'})
        replies=list(PromptFaceService(fast_prompts=True).StreamSession(iter([req]),Mock()))
        self.assertEqual(replies[0].type,pb.ERROR)
