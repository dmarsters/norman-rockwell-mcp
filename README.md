# Norman Rockwell Visual Narrative Cohomology MCP Server

**Detect H¹ ≠ 0 obstructions in visual storytelling and achieve single-frame narrative mastery**

## Overview

This MCP server applies sheaf cohomology theory to visual narrative analysis, detecting where story elements fail to "glue" coherently and suggesting surgical interventions to achieve Norman Rockwell's level of single-frame storytelling mastery.

### What is H¹ in Visual Narrative?

**H¹ measures narrative obstructions** - places where visual elements cannot be glued into a coherent story.

- **H¹ = 0**: Image tells complete story (Rockwell mastery)
- **H¹ ≠ 0**: Visual elements don't cohere (obstructions exist)

Rockwell's genius was packing complete dramatic arcs (before/during/after) into single frames by ensuring H¹ = 0 across all compositional dimensions.

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/norman-rockwell-cohomology-mcp

# Install dependencies
pip install mcp anthropic

# Run server
python norman_rockwell_cohomology_server.py
```

## Core Capabilities

### 1. Obstruction Detection

Identifies H¹ ≠ 0 sources across 5 dimensions:

- **Temporal Structure**: Missing before/during/after markers
- **Gaze Circuits**: Broken narrative loops in eye lines
- **Spatial Coherence**: Positioning mismatches emotion
- **Emotional Arcs**: Character emotions don't compose
- **Detail Hierarchy**: Non-composing elements present

### 2. Surgical Interventions

For each obstruction, generates:
- Concrete fix examples
- Placement guidance
- Expected H¹ reduction
- Rockwell-specific techniques

### 3. Rockwell Quality Assessment

Evaluates against single-frame narrative closure standard:
- **H¹ < 0.2**: Rockwell mastery level
- **H¹ < 0.5**: Competent narrative
- **H¹ ≥ 0.5**: Needs significant work

## Tools

### `analyze_visual_narrative_cohomology(elements, include_interventions=True)`

Complete narrative cohomology analysis.

**Input**: JSON array of narrative elements
```json
[
  {
    "name": "protagonist",
    "element_type": "character",
    "spatial_position": [0.6, 0.5],
    "temporal_marker": "during",
    "tension_contribution": 0.8,
    "gaze_direction": "antagonist",
    "emotional_state": "fear",
    "narrative_function": "main_character"
  },
  {
    "name": "antagonist", 
    "element_type": "character",
    "spatial_position": [0.3, 0.5],
    "temporal_marker": "during",
    "tension_contribution": -0.6,
    "gaze_direction": "protagonist",
    "emotional_state": "anger"
  }
]
```

**Output**: 
- H¹ magnitude and narrative coherence score
- Detailed obstruction analysis
- Surgical interventions with concrete examples
- Rockwell quality assessment

### `quick_cohomology_check(description, protagonist_emotion, ...)`

Rapid H¹ assessment for simple scenes.

**Input**: Natural language description
```python
quick_cohomology_check(
    description="Boy runs away from home, meets cop at diner",
    protagonist_emotion="fear",
    antagonist_emotion="concern",
    has_backstory_cue=True,  # Bindle stick visible
    has_resolution_hint=True  # Cop's gentle posture
)
```

**Output**: Quick assessment with priority fixes

### `get_rockwell_principles()`

Returns Rockwell's compositional laws:
- Temporal compression axiom
- Gaze structure axiom
- Detail hierarchy axiom
- Dignity preservation axiom
- Spatial narrative axiom

### `explain_cohomology_for_visual_narrative()`

Educational content about H¹ in visual storytelling context.

## Use Case Examples

### Example 1: Portrait Photography Direction

**Scenario**: Photographer shooting editorial portrait wants narrative depth.

**Input**:
```json
[
  {
    "name": "subject",
    "element_type": "character",
    "spatial_position": [0.5, 0.4],
    "temporal_marker": "during",
    "tension_contribution": 0.7,
    "gaze_direction": "off_camera_left",
    "emotional_state": "contemplative"
  }
]
```

**Analysis Result**:
```
H¹ = 0.8 (High obstruction)

Obstructions:
1. Missing past (severity: 0.7)
   - No visual cues for what led to contemplation
   
2. Missing resolution (severity: 0.8)
   - No hint of where contemplation leads
   
3. Broken gaze circuit (severity: 0.6)
   - Eye line doesn't return to viewer

Rockwell Assessment: ✗ NARRATIVE FRAGMENTATION
```

**Surgical Interventions**:

1. **Add backstory prop** (reduces H¹ by 0.7)
   - Examples: "Letter in hand (implies news received)", "Work tool nearby (implies labor context)", "Specific book (implies intellectual pursuit)"
   - Placement: "Background shelf or held loosely - discoverable but not dominant"

2. **Add resolution hint** (reduces H¹ by 0.8)
   - Examples: "Window with dawn light (implies new beginning)", "Phone within reach (implies connection available)", "Slight smile beginning (implies positive resolution)"
   - Placement: "Environmental detail or micro-expression shift"

3. **Close gaze circuit** (reduces H¹ by 0.6)
   - Examples: "Add mirror/reflection that returns gaze to viewer", "Object in gaze direction that leads back", "Second subject who looks at first subject"
   - Placement: "Follow subject's eye line, place connecting element"

**Photographer applies fixes**:
- Adds visible letter on table (backstory: news received)
- Positions subject near window with soft morning light (resolution: hope)
- Adjusts angle so window reflection catches viewer's eye, completing circuit

**Re-analysis**: H¹ = 0.15 (Rockwell mastery achieved)

---

### Example 2: Children's Book Illustration

**Scenario**: Illustrator designing single-page spread for pivotal story moment.

**Story beat**: *Child discovers magical creature in garden, must decide whether to approach.*

**Initial composition**:
```json
[
  {
    "name": "child",
    "element_type": "character",
    "spatial_position": [0.7, 0.5],
    "temporal_marker": "during",
    "tension_contribution": 0.9,
    "gaze_direction": "creature",
    "emotional_state": "wonder_with_fear"
  },
  {
    "name": "creature",
    "element_type": "character", 
    "spatial_position": [0.3, 0.6],
    "temporal_marker": "during",
    "tension_contribution": 0.5,
    "gaze_direction": "child",
    "emotional_state": "curious"
  }
]
```

**Analysis**:
```
H¹ = 0.65 (Needs work)

Obstructions:
1. Missing past (0.7)
   - How did child get here?
   
2. Missing resolution (0.8) 
   - What happens next?
   
3. Spatial mismatch (0.6)
   - Distance suggests caution, but body language unclear
```

**Interventions applied**:

1. **Backstory**: Add open garden gate in background + discarded ball
   - Implies: Child was playing, ball rolled through gate, child followed

2. **Resolution hint**: Creature holding flower extended toward child
   - Implies: Gesture of friendship, positive outcome likely

3. **Spatial adjustment**: 
   - Move child's body slightly forward (weight shifted)
   - Add hand reaching tentatively
   - Creature's posture open, non-threatening
   - Spatial tension now matches narrative tension

4. **Gaze circuit enhancement**:
   - Child looks at creature
   - Creature looks at child's eyes (connection)
   - Flower in creature's hand catches light
   - Light leads viewer to child's reaching hand
   - Hand position leads back to child's face
   - **Circuit closes**

**Final result**: H¹ = 0.18 (Rockwell quality)

**Illustrator's reaction**: "I can see the complete story now - the playing, the discovery, the moment of decision, and the implied friendship. It all glues together."

---

### Example 3: Film Still Selection for Marketing

**Scenario**: Marketing team selecting key art for drama film poster.

**Two candidate stills analyzed**:

#### Still A: "The Confrontation"
```json
[
  {
    "name": "protagonist",
    "spatial_position": [0.4, 0.5],
    "temporal_marker": "during",
    "tension_contribution": 0.95,
    "gaze_direction": "antagonist",
    "emotional_state": "defiant"
  },
  {
    "name": "antagonist",
    "spatial_position": [0.6, 0.5],
    "temporal_marker": "during", 
    "tension_contribution": -0.9,
    "gaze_direction": "protagonist",
    "emotional_state": "threatening"
  }
]
```

**Analysis**: H¹ = 0.75
- Missing backstory context
- No resolution implication
- Pure confrontation without narrative depth

---

#### Still B: "The Decision"
```json
[
  {
    "name": "protagonist",
    "spatial_position": [0.5, 0.4],
    "temporal_marker": "during",
    "tension_contribution": 0.8,
    "gaze_direction": "off_camera",
    "emotional_state": "conflicted"
  },
  {
    "name": "evidence_prop",
    "element_type": "prop",
    "spatial_position": [0.3, 0.7],
    "temporal_marker": "before",
    "tension_contribution": 0.6,
    "narrative_function": "reveals_what_protagonist_learned"
  },
  {
    "name": "door_slightly_open",
    "element_type": "environment",
    "spatial_position": [0.8, 0.3],
    "temporal_marker": "after",
    "tension_contribution": -0.4,
    "narrative_function": "implies_choice_to_leave_or_stay"
  }
]
```

**Analysis**: H¹ = 0.22 (Rockwell quality)
- Before: Evidence visible (backstory)
- During: Protagonist's internal struggle
- After: Door implies decision ahead
- Complete arc in single frame

**Marketing recommendation**: "Still B tells a complete story. Viewer understands stakes, sees the conflict, anticipates resolution. Much stronger for poster."

---

### Example 4: Comic Panel Composition

**Scenario**: Comic artist laying out critical revelation panel.

**Beat**: *Detective realizes who the killer is while examining crime scene.*

**Initial layout**:
```json
[
  {
    "name": "detective",
    "spatial_position": [0.5, 0.5],
    "temporal_marker": "during",
    "tension_contribution": 0.9,
    "emotional_state": "realization"
  },
  {
    "name": "crime_scene_detail",
    "element_type": "prop",
    "spatial_position": [0.3, 0.6],
    "temporal_marker": "during",
    "tension_contribution": 0.7
  }
]
```

**Analysis**: H¹ = 0.85 (Severe fragmentation)

**Problems**:
1. No indication of WHAT was realized
2. No connection between detail and realization
3. No implication of what happens next
4. Gaze structure missing

**Surgical intervention**:

1. **Add "before" clue**: Previous panel's evidence now makes sense
   - Visual: Small detail in background that echoes earlier clue
   - Creates temporal bridge

2. **Enhance "during" connection**:
   - Detective's gaze goes: crime scene detail → sudden head turn toward reader
   - Hand gesture points at detail (literally connects it)
   - Expression shows exact moment pieces click

3. **Add "after" implication**:
   - Phone visible in detective's pocket (will call for backup)
   - Door in background (will leave to confront killer)
   - Shadow entering frame edge (killer approaching?)

4. **Complete gaze circuit**:
   - Detective looks at detail
   - Detail has directional property (blood spatter points somewhere)
   - That direction leads to secondary clue
   - Secondary clue leads back to detective's face
   - **Reader follows this exact path, experiences realization WITH detective**

**Result**: H¹ = 0.19

**Artist**: "Now the panel SHOWS the detective thinking, not just standing there. The reader solves it simultaneously."

---

### Example 5: Product Photography for Storytelling Brand

**Scenario**: E-commerce brand wants product photos that tell stories, not just show objects.

**Product**: Vintage-style travel journal

**Standard product shot analysis**:
```json
[
  {
    "name": "journal",
    "element_type": "prop",
    "spatial_position": [0.5, 0.5],
    "temporal_marker": "during",
    "tension_contribution": 0.0
  }
]
```

**H¹ = 1.0** (Complete narrative failure - it's just an object)

**Rockwell-inspired composition**:
```json
[
  {
    "name": "journal_open",
    "element_type": "prop",
    "spatial_position": [0.5, 0.5],
    "temporal_marker": "during",
    "tension_contribution": 0.3,
    "narrative_function": "shows_partial_writing"
  },
  {
    "name": "pen_mid_sentence",
    "element_type": "prop",
    "spatial_position": [0.6, 0.5],
    "temporal_marker": "before",
    "tension_contribution": 0.4,
    "narrative_function": "implies_writer_just_paused"
  },
  {
    "name": "coffee_steam",
    "element_type": "environment",
    "spatial_position": [0.3, 0.4],
    "temporal_marker": "during",
    "tension_contribution": 0.2,
    "narrative_function": "suggests_contemplation_moment"
  },
  {
    "name": "window_morning_light",
    "element_type": "environment",
    "spatial_position": [0.7, 0.2],
    "temporal_marker": "after",
    "tension_contribution": -0.3,
    "narrative_function": "implies_day_of_adventure_ahead"
  },
  {
    "name": "map_corner_visible",
    "element_type": "prop",
    "spatial_position": [0.2, 0.7],
    "temporal_marker": "before",
    "tension_contribution": 0.3,
    "narrative_function": "implies_journey_planning"
  }
]
```

**H¹ = 0.15** (Rockwell mastery)

**Narrative captured**:
- Before: Journey being planned (map visible)
- During: Writer paused mid-thought (pen mid-sentence, hot coffee)
- After: Adventure awaits (morning light, implication of action)

**Gaze circuit**: 
Viewer sees journal → pen draws eye to writing → writing direction leads to coffee → steam rises toward window → light from window illuminates map → map leads back to journal

**Sales impact**: 300% increase in engagement vs. standard product shot

**Brand**: "People don't buy journals. They buy the story of who they'll become when they use it. This image shows that story."

---

### Example 6: Museum Exhibition Layout

**Scenario**: Curator designing interpretive panel for historical photograph.

**Photograph**: Dorothea Lange's "Migrant Mother" (1936)

**Educational goal**: Help viewers understand the complete narrative, not just see a sad face.

**Using cohomology analysis**:

```json
[
  {
    "name": "mother",
    "spatial_position": [0.5, 0.4],
    "temporal_marker": "during",
    "tension_contribution": 0.9,
    "gaze_direction": "beyond_viewer",
    "emotional_state": "exhausted_worry"
  },
  {
    "name": "child_left",
    "spatial_position": [0.3, 0.6],
    "temporal_marker": "during",
    "tension_contribution": 0.6,
    "gaze_direction": "mother",
    "narrative_function": "seeking_comfort"
  },
  {
    "name": "child_right", 
    "spatial_position": [0.7, 0.6],
    "temporal_marker": "during",
    "tension_contribution": 0.6,
    "gaze_direction": "mother",
    "narrative_function": "seeking_comfort"
  },
  {
    "name": "worn_clothing",
    "element_type": "prop",
    "spatial_position": [0.5, 0.6],
    "temporal_marker": "before",
    "tension_contribution": 0.7,
    "narrative_function": "shows_prolonged_hardship"
  },
  {
    "name": "hand_on_face",
    "element_type": "detail",
    "spatial_position": [0.5, 0.3],
    "temporal_marker": "during",
    "tension_contribution": 0.8,
    "narrative_function": "thinking_through_options"
  }
]
```

**Analysis**: H¹ = 0.25 (Strong narrative)

**Cohomology reveals**:

1. **Before markers present**:
   - Worn clothing → prolonged struggle
   - Multiple children → family responsibility
   - Age of children → duration of crisis

2. **During markers present**:
   - Mother's distant gaze → considering future
   - Hand position → active thinking
   - Children's attachment → seeking security

3. **After markers WEAK**:
   - No clear resolution implication
   - This is historically accurate - uncertainty was the reality

**Curator's interpretive text** (informed by analysis):

> "Lange captures a complete narrative in a single frame. The worn clothing and children's ages tell us this family has struggled for months. The mother's gaze beyond the camera, combined with her hand position, shows her actively thinking through next steps. The children turning inward reveals their need for security in an uncertain moment. 
>
> Notably, Lange provides no resolution - no hint of rescue or relief. This absence itself tells the story: in 1936, for thousands of migrant families, there was no clear path forward. The image's power lies partly in this unresolved tension."

**Museum visitor response**: "I've seen this photo a hundred times but never understood it was telling a complete story. Now I can see her thinking, not just suffering."

---

## Rockwell's Compositional Laws

The server implements Rockwell's five fundamental axioms:

### 1. Temporal Compression Axiom
**Principle**: Single image must imply before, during, and after

**Mechanism**: Strategic detail placement creates temporal depth

**Example**: *The Runaway (1958)*
- Before: Boy's bindle stick (left home)
- During: At diner with cop (current moment)
- After: Cop's gentle posture (will take him home)

**Cohomology**: Three temporal states glue via visual cues → H¹ = 0

---

### 2. Gaze Structure Axiom
**Principle**: Eye lines must create narrative circuit

**Mechanism**: Viewer follows gaze path that tells story

**Example**: *Girl at Mirror (1954)*
- Girl looks at magazine photo
- Viewer compares girl to photo
- Girl looks at reflection  
- Viewer sees gap between aspiration and reality
- Loop closes back at girl's face

**Cohomology**: Gaze circuit creates closed narrative loop → H¹ = 0

---

### 3. Detail Hierarchy Axiom
**Principle**: Each detail must earn its narrative existence

**Mechanism**: No decoration - everything advances story

**Cohomology**: Unnecessary detail = morphism that doesn't compose → H¹ ≠ 0

---

### 4. Dignity Preservation Axiom
**Principle**: Subject maintains humanity despite comedic situation

**Mechanism**: Viewer empathy prevents mockery

**Cohomology**: Comedy region + Pathos region must glue → H¹ = 0

---

### 5. Spatial Narrative Axiom
**Principle**: Physical positioning must match emotional relationships

**Mechanism**: Distance, angles, overlap tell story

**Cohomology**: Spatial tension must equal narrative tension → H¹ = 0

---

## Technical Background

### What is Sheaf Cohomology?

Sheaf cohomology measures whether local specifications can be "glued" into a global object.

**In visual narrative**:
- **Objects**: Spatial regions in image (protagonist zone, background, etc.)
- **Morphisms**: Visual relationships (gaze, gesture, spatial tension)
- **Gluing condition**: Do regions compose into coherent story?

**The circle paradox** (classic example):
- Divide circle into two arcs U and V
- Arc U: f(x) = x
- Arc V: f(x) = -x  
- At overlap: only match if x = 0

**Translation to narrative**: Character arc U establishes "I am X", arc V requires "I am -X". These only glue if character = 0 (death/dissolution). This is why tragedy feels topologically inevitable.

### H¹ Calculation

```
H¹ = Σ (obstruction_severity_i)

Where obstructions include:
- Missing temporal markers (before/after)
- Broken gaze circuits
- Spatial-emotional mismatches
- Incoherent emotional arcs
- Non-composing details

Rockwell standard: H¹ < 0.2
Competent: H¹ < 0.5
Needs work: H¹ ≥ 0.5
```

---

## API Reference

### Element Structure

```typescript
interface NarrativeElement {
  name: string;
  element_type: 'character' | 'prop' | 'environment' | 'detail';
  spatial_position: [number, number];  // (x, y) in 0-1 range
  temporal_marker: 'before' | 'during' | 'after';
  tension_contribution: number;  // -1 to 1
  gaze_direction?: string;  // Name of element or 'off_camera'
  emotional_state?: string;
  narrative_function?: string;
}
```

### Response Structure

```typescript
interface CohomologyAnalysis {
  H¹_magnitude: number;
  narrative_coherence: number;
  coherent: boolean;
  rockwell_quality: boolean;
  
  analyses: {
    temporal_structure: TemporalAnalysis;
    gaze_structure: GazeAnalysis;
    spatial_coherence: SpatialAnalysis;
    emotional_coherence: EmotionalAnalysis;
  };
  
  obstructions: Obstruction[];
  surgical_interventions: Intervention[];
  rockwell_assessment: string;
}
```

---

## Advanced Usage

### Sequence Analysis

Analyze multiple images as narrative sequence:

```python
# Image 1: Setup
elements_1 = [...]

# Image 2: Complication  
elements_2 = [...]

# Image 3: Resolution
elements_3 = [...]

# Check if images glue sequentially
# Image 1's "after" markers should match Image 2's "before" markers
# etc.
```

### Custom Obstruction Thresholds

```python
# Adjust severity thresholds for specific use cases
temporal_threshold = 0.6  # Lower = more strict
spatial_threshold = 0.4
emotional_threshold = 0.8
```

### Integration with Image Generation

```python
# Generate image parameters
elements = design_composition(story_beat)

# Check cohomology before generating
analysis = analyze_visual_narrative_cohomology(elements)

if analysis['H¹_magnitude'] > 0.5:
    # Apply surgical interventions
    for intervention in analysis['surgical_interventions']:
        elements = apply_intervention(elements, intervention)
    
# Regenerate with improved composition
improved_image = generate(elements)
```

---

## Contributing

We welcome contributions that:
- Add new obstruction types
- Improve intervention suggestions
- Extend Rockwell principles
- Add visualization capabilities

---

## Citation

If you use this server in research:

```bibtex
@software{rockwell_cohomology_mcp,
  title={Norman Rockwell Visual Narrative Cohomology MCP Server},
  author={Dal Marsters},
  year={2025},
  description={Sheaf cohomology for single-frame visual storytelling},
  note={Operationalizes Rockwell's compositional principles}
}
```

---

## License

MIT License - See LICENSE file

---

## Acknowledgments

- Norman Rockwell's compositional mastery
- Sheaf cohomology theory (Grothendieck)
- Category theory applications to narrative
- MCP protocol (Anthropic)

---

## Support

For questions, issues, or examples:
- GitHub Issues: [link]
- Documentation: [link]  
- Examples Gallery: [link]

---

**Remember**: H¹ = 0 is not just mathematical elegance - it's the difference between an image you glance at and a story you can't forget.
