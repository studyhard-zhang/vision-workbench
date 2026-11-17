import unittest,copy
from app.small_targets import analyze

class SmallTargetTests(unittest.TestCase):
    def test_training_projection_uses_current_roi_dimensions_and_ignores_validation_targets(self):
        data={'images':[{'width':2000,'height':1000,'subset':'train','boxes':[{'width':4,'height':100}]},{'width':2000,'height':1000,'subset':'train','boxes':[]},{'width':20,'height':20,'subset':'val','boxes':[{'width':20,'height':20}]}]};before=copy.deepcopy(data)
        result=analyze(data,640,'segment');self.assertEqual(result['target_count'],1);self.assertEqual(result['current']['p10_short_pixels'],1.28);self.assertEqual(result['current']['under4'],1);self.assertEqual(result['negative_images'],1);self.assertEqual(result['thin_box_count'],1);self.assertTrue(result['size_limit_insufficient']);self.assertIn('mask_ratio=4',' '.join(result['guidance']));self.assertEqual(data,before)
        data['images'][0]['width']=500;data['images'][0]['height']=500;self.assertEqual(analyze(data,640)['current']['p10_short_pixels'],5.12)
    def test_empty_positive_set_has_no_fake_size_or_parameter_guarantee(self):
        result=analyze({'images':[{'width':1000,'height':500,'subset':'train','boxes':[]}]},640)
        self.assertIsNone(result['suggested_imgsz']);self.assertIsNone(result['current']['p10_short_pixels']);self.assertEqual(result['current']['under4'],0)
    def test_every_supported_size_keeps_counts_tied_to_that_size_and_never_changes_training_config(self):
        data={'images':[{'width':1024,'height':1024,'subset':'train','boxes':[{'width':8,'height':8},{'width':16,'height':4}]}]}
        result=analyze(data,1024,adamw_verified=False);rows={row['imgsz']:row for row in result['sizes']};self.assertEqual(rows[1024]['under4'],0);self.assertEqual(rows[512]['under4'],1);self.assertIn('未开放显式AdamW',' '.join(result['guidance']));self.assertEqual(result['suggested_imgsz'],2048)
