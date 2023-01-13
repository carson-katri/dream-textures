from collections import namedtuple

PromptToken = namedtuple('PromptToken', ['id', 'label', 'values'])
PromptStructure = namedtuple('PromptStructure', ['id', 'label', 'structure', 'generate'])

framing_token = PromptToken('framing', 'Framing', (
    ('ecu', 'Extreme Close-up'),
    ('cu', 'Close-up'),
    ('mcu', 'Medium Close Up'),
    ('ms', 'Medium Shot'),
    ('ls', 'Long Shot'),
    ('els', 'Extra Long Shot'),
))

position_token = PromptToken('position', 'Position', (
    ('overhead', 'Overhead View'),
    ('aerial', 'Aerial View'),
    ('low', 'Low Angle'),
    ('dutch', 'Dutch Angle'),
    ('ots', 'Over-the-shoulder shot'),
))

film_type_token = PromptToken('film_type', 'Film Type', (
    ('bw', 'Black & White'),
    ('fc', 'Full Color'),
    ('cine', 'Cinematic'),
    ('polaroid', 'Polaroid'),
    ('anaglyph', 'Anaglyph'),
    ('double', 'Double Exposure'),
))

camera_settings_token = PromptToken('camera_settings', 'Camera Settings', (
    ('high_speed', 'Fast Shutter Speed'),
    ('long_exposure', 'Long Exposure'),
    ('bokeh', 'Shallow Depth of Field'),
    ('deep_dof', 'Deep Depth of Field'),
    ('tilt_shift', 'Tilt Shift'),
    ('motion_blur', 'Motion Blur'),
    ('telephoto', 'Telephoto Lens'),
    ('macro', 'Macro Lens'),
    ('wide_angle', 'Wide Angle Lens'),
    ('fish_eye', 'Fish-Eye Lens'),
))

shooting_context_token = PromptToken('shooting_context', 'Shooting Context', (
    ('film_still', 'Film Still'),
    ('photograph', 'Photograph'),
    ('studio_portrait', 'Studio Portrait Photograph'),
    ('outdoor', 'Outdoor Photograph'),
    ('cctv', 'Surveillance Footage'),
))

subject_token = PromptToken('subject', 'Subject', ())

lighting_token = PromptToken('lighting', 'Lighting', (
    ('golden_hour', 'Golden Hour'),
    ('blur_hour', 'Blue Hour'),
    ('midday', 'Midday'),
    ('overcast', 'Overcast'),
    ('silhouette', 'Mostly Silhouetted'),
    
    ('warm', 'Warm Lighting, 2700K'),
    ('cold', 'Flourescent Lighting, 4800K'),
    ('flash', 'Harsh Flash'),
    ('ambient', 'Ambient Lighting'),
    ('dramatic', 'Dramatic Lighting'),
    ('backlit', 'Backlit'),
    ('studio', 'Studio Lighting'),
    ('above', 'Lit from Above'),
    ('below', 'Lit from Below'),
    ('left', 'Lit from the Left'),
    ('right', 'Lit from the Right'),
))

def texture_prompt(tokens):
    return f"{tokens.subject} texture"
texture_structure = PromptStructure(
    'texture',
    'Texture',
    [subject_token],
    texture_prompt
)

def photography_prompt(tokens):
    return f"A {tokens.framing} {tokens.position} {tokens.film_type} {tokens.camera_settings} {tokens.shooting_context} of {tokens.subject}, {tokens.lighting}"

photography_structure = PromptStructure(
    'photography',
    'Photography',
    (subject_token, framing_token, position_token, film_type_token, camera_settings_token, shooting_context_token, lighting_token),
    photography_prompt
)

subject_type_token = PromptToken('subject_type', 'Subject Type', (
    ('environment', 'Environment'),
    ('character', 'Character'),
    ('weapon', 'Weapon'),
    ('vehicle', 'Vehicle'),
))

genre_token = PromptToken('genre', 'Genre', (
    ('scifi', 'Sci-Fi'),
    ('fantasy', 'Fantasy'),
    ('cyberpunk', 'Cyberpunk'),
    ('cinematic', 'Cinematic'),
))

def concept_art_prompt(tokens):
    return f"{tokens.subject}, {tokens.subject_type} concept art, {tokens.genre} digital painting, trending on ArtStation"

concept_art_structure = PromptStructure(
    'concept_art',
    'Concept Art',
    (subject_token, subject_type_token, genre_token),
    concept_art_prompt
)

def custom_prompt(tokens):
    return f"{tokens.subject}"
custom_structure = PromptStructure(
    'custom',
    'Custom',
    [subject_token],
    custom_prompt
)

def file_batch_prompt(tokens):
    return f""
file_batch_structure = PromptStructure(
    'file_batch',
    "File Batch",
    [],
    file_batch_prompt
)

prompt_structures = [
    custom_structure,
    texture_structure,
    photography_structure,
    concept_art_structure,
    file_batch_structure
]

def map_structure(x):
    return (x.id, x.label, '')
prompt_structures_items = list(map(map_structure, prompt_structures))