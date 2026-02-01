import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Dictionaries to store loaded templates
ALL_TEMPLATES = {}

try: from caltech101 import CALTECH101_TEMPLATES; ALL_TEMPLATES['caltech101'] = CALTECH101_TEMPLATES
except ImportError: pass

try: from dtd import DTD_TEMPLATES; ALL_TEMPLATES['describabletextures'] = DTD_TEMPLATES
except ImportError: pass

try: from eurosat import EUROSAT_TEMPLATES; ALL_TEMPLATES['eurosat'] = EUROSAT_TEMPLATES
except ImportError: pass

try: from fgvc_aircraft import FGVC_AIRCRAFT_TEMPLATES; ALL_TEMPLATES['fgvcaircraft'] = FGVC_AIRCRAFT_TEMPLATES
except ImportError: pass

try: from food101 import FOOD101_TEMPLATES; ALL_TEMPLATES['food101'] = FOOD101_TEMPLATES
except ImportError: pass

try: from oxford_flowers import OXFORD_FLOWERS_TEMPLATES; ALL_TEMPLATES['oxfordflowers'] = OXFORD_FLOWERS_TEMPLATES
except ImportError: pass

try: from oxford_pets import OXFORD_PETS_TEMPLATES; ALL_TEMPLATES['oxfordpets'] = OXFORD_PETS_TEMPLATES
except ImportError: pass

try: from stanford_cars import STANFORD_CARS_TEMPLATES; ALL_TEMPLATES['stanfordcars'] = STANFORD_CARS_TEMPLATES
except ImportError: pass

try: from sun397 import SUN397_TEMPLATES; ALL_TEMPLATES['sun397'] = SUN397_TEMPLATES
except ImportError: pass

try: from ucf101 import UCF101_TEMPL
ATES; ALL_TEMPLATES['ucf101'] = UCF101_TEMPLATES
except ImportError: pass

try: from imagenet import IMAGENET_TEMPLATES; ALL_TEMPLATES['imagenet'] = IMAGENET_TEMPLATES
except ImportError: pass

def load_CuPL_templates(dataset_name):
    # Normalize name to match keys
    key = dataset_name.lower().replace("_", "")
    
    # Check for direct match
    if key in ALL_TEMPLATES:
        return ALL_TEMPLATES[key]
    
    # Partial match check (e.g. 'Caltech101' matching 'caltech101')
    for k, v in ALL_TEMPLATES.items():
        if k in key or key in k:
            return v
            
    print(f"Warning: No GPT-3 templates found for {dataset_name}. Using empty dict.")
    return {}