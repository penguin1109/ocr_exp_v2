INFO={"annotations": [{"image": 0, "text": "testing"}]}
import json
with open('/home/guest/ocr_exp_v2/data/croped_outdoor.json', 'w') as f:
    json.dump(INFO,f, indent=4)