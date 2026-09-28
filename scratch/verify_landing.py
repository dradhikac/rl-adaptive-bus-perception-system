import urllib.request

html = urllib.request.urlopen('http://127.0.0.1:5000/').read().decode()

# 1. Navigation checks
assert 'Live Demo</a></li>' not in html, 'Live Demo must not be in navbar list'
assert 'data-section="overview"' in html, 'Overview link missing'
assert 'data-section="technology"' in html, 'Technology link missing'
assert 'data-section="features"' in html, 'Features link missing'
assert 'data-section="architecture"' in html, 'Architecture link missing'
assert 'btn-nav-demo' in html and 'View Demo' in html, 'View Demo CTA missing'

# 2. Technology 6 core modules checks
assert '360° Multi-Camera Perception' in html, '360 Perception missing'
assert 'Traffic Signal Recognition' in html, 'Traffic signal missing'
assert 'Blind Spot Intelligence' in html, 'Blind spot missing'
assert 'Adaptive Sensor Fusion' in html, 'Sensor fusion missing'
assert 'RL Decision Engine' in html, 'RL Engine missing'
assert 'Decision Fusion Layer' in html, 'Decision Fusion Layer missing'

# 3. Architecture 5 stages checks
assert 'STAGE 01' in html and 'PERCEPTION' in html
assert 'STAGE 02' in html and 'SENSOR INTELLIGENCE' in html
assert 'STAGE 03' in html and 'RISK & SAFETY' in html
assert 'STAGE 04' in html and 'DECISION' in html
assert 'STAGE 05' in html and 'ACTION' in html

print("SUCCESS: All landing page structure, navigation, technology, features, and architecture assertions PASSED!")
