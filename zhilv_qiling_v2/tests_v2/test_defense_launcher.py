"""Explicit GPU flags leave the existing CPU default and process ownership intact."""
import os
import sys
import types
from unittest.mock import Mock, patch
from zhilv_qiling_v2.tests_v2.test_start_web import LauncherTest
import start_web as W


class DefenseLauncherTest(LauncherTest):
    def test_explicit_offline_mode_does_not_connect_to_a_resident_model(self):
        self.args.with_model = False
        with patch.object(W, 'local_model_status') as status:
            self.assertIsNone(W.connect_local_model(self.args))
        status.assert_not_called()
        self.assertEqual(os.environ['ZLQL_ENABLE_LLM'], '0')

    def test_resident_mode_retains_model_after_web_shutdown(self):
        child = self.child()
        server = Mock()
        server.serve_forever.side_effect = KeyboardInterrupt
        with patch.object(W, 'workspace_status', return_value=None), \
             patch.object(W, 'connect_local_model', return_value=child) as connect, \
             patch.object(W.sys, 'argv', ['start_web.py', '--keep-model-running', '--no-browser']), \
             patch.object(sys.modules['zhilv_qiling_v2'], 'api', types.SimpleNamespace(serve=Mock(return_value=server)), create=True):
            self.assertEqual(W.main(), 0)
        self.assertTrue(connect.call_args.args[0].with_model)
        server.server_close.assert_called_once()
        child.terminate.assert_not_called()

    def test_resident_mode_still_cleans_up_failed_web_initialization(self):
        child = self.child()
        with patch.object(W, 'workspace_status', return_value=None), \
             patch.object(W, 'connect_local_model', return_value=child), \
             patch.object(W.sys, 'argv', ['start_web.py', '--keep-model-running', '--no-browser']), \
             patch.object(sys.modules['zhilv_qiling_v2'], 'api', types.SimpleNamespace(serve=Mock(side_effect=RuntimeError('mock failed startup'))), create=True):
            self.assertEqual(W.main(), 1)
        child.terminate.assert_called_once()

    def test_explicit_gpu_offload_and_cache_flags(self):
        binary, model = self.assets()
        child = self.child()
        self.args.model_gpu_layers=99
        self.args.model_cache_type='q8_0'
        self.args.model_context_tokens=16384
        with patch.object(W,'local_model_status',side_effect=[None, self.native()]), \
             patch.object(W.subprocess,'Popen',return_value=child) as launch:
            self.assertIs(W.connect_local_model(self.args),child)
        args=launch.call_args.args[0]
        self.assertEqual(args[args.index('--n-gpu-layers')+1],'99')
        self.assertEqual(args[args.index('--cache-type-k')+1],'q8_0')
        self.assertEqual(args[args.index('--cache-type-v')+1],'q8_0')
        self.assertEqual(args[args.index('--ctx-size')+1],'16384')


if __name__=='__main__':
    import unittest
    unittest.main()
