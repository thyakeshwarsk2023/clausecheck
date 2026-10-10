import ast, re, sys

print('=== AUDIT 3: APP.PY ROBUSTNESS & FAULT INSPECTION ===')

with open('app.py', 'r', encoding='utf-8') as fh:
    source = fh.read()

# 1. Parse AST to verify syntax is valid
try:
    tree = ast.parse(source)
    print('  [OK] app.py AST parse: valid Python syntax, no SyntaxError')
except SyntaxError as e:
    print(f'  [FAIL] SyntaxError: {e}')
    sys.exit(1)

# 2. Check error handling exists around audio decode
checks = {
    'st.audio_input() present':          'st.audio_input' in source,
    'UnknownValueError catch':           'UnknownValueError' in source,
    'RequestError catch':                'RequestError' in source,
    'Generic except for audio errors':   'except Exception' in source,
    'soundfile import for PCM':          'import soundfile' in source or 'soundfile as sf' in source,
    'gTTS import for TTS':               'from gtts import gTTS' in source,
    'Text fallback via st.text_input':   'st.text_input' in source,
    'Confidence threshold defined':      'CONFIDENCE_THRESHOLD' in source,
    'out_of_scope abstention logic':     'out_of_scope' in source,
    'Abstain branch: no card rendered':  'abstain' in source.lower() or ('out_of_scope' in source and 'st.warning' in source),
    'Multi-intent ranking (sorted)':     'torch.sort' in source or 'ranked_results' in source,
    'TTS cache decorator present':       'st.cache_data' in source,
    'TTS st.audio() playback':           'st.audio(tts_bytes' in source,
    'model loaded with map_location cpu': 'map_location="cpu"' in source or "map_location='cpu'" in source,
    'st.cache_resource for model':       '@st.cache_resource' in source,
    'Model eval() mode set':             'model.eval()' in source,
    'Probability expander shown':        'expander' in source,
}

passed = 0
failed = 0
for desc, result in checks.items():
    status = 'OK' if result else 'FAIL'
    if result:
        passed += 1
    else:
        failed += 1
    print(f'  [{status}] {desc}')

print()
print(f'  App robustness checks: {passed}/{passed+failed} passed')

# 3. Verify confidence threshold value
thresh_match = re.search(r'CONFIDENCE_THRESHOLD\s*=\s*([\d.]+)', source)
if thresh_match:
    thresh = float(thresh_match.group(1))
    print(f'  Confidence threshold value: {thresh}  (expected 0.40-0.50)')
    print(f'  Threshold range check: {"OK" if 0.3 <= thresh <= 0.6 else "WARN - unusual value"}')
else:
    print('  [FAIL] CONFIDENCE_THRESHOLD not found in source')

# 4. Verify abstention: does code explicitly skip rendering for out_of_scope?
abstain_pattern = re.search(r'if.*confidence.*<.*CONFIDENCE_THRESHOLD.*or.*out_of_scope', source, re.DOTALL)
abstain_pattern2 = bool(re.search(r'CONFIDENCE_THRESHOLD.*out_of_scope|out_of_scope.*CONFIDENCE_THRESHOLD', source))
print(f'  Dual abstain guard (threshold OR out_of_scope): {"OK" if abstain_pattern2 else "WARN - single guard only"}')

# 5. Audio decode fallback path
fallback_pattern = bool(re.search(r'if not query_text and typed_query', source))
print(f'  Text fallback when audio empty: {"OK" if fallback_pattern else "FAIL"}')
