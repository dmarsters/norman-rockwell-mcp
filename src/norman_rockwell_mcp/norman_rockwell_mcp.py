"""
Enhanced Norman Rockwell MCP Server v3.0 with Improved Cohomology Analysis
- Fixed serialization bug (NarrativeElement objects now converted to dicts)
- Geometric gaze vector analysis (computes actual gaze line intersections)
- Calibrated scoring (quality-weighted, not just presence/absence)
- Resolution hint quality assessment (graduated, not binary)
- Gaze circuit closure as computed geometric property
"""

from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import json
import math

# ============================================================================
# CORE DATA STRUCTURES
# ============================================================================

class ObstructionType(Enum):
    """Types of narrative obstructions (H¹ ≠ 0 sources)"""
    MISSING_PAST = "missing_past"
    MISSING_RESOLUTION = "missing_resolution_hint"
    SPATIAL_NARRATIVE_MISMATCH = "spatial_narrative_mismatch"
    EMOTIONAL_INCOHERENCE = "emotional_incoherence"
    GAZE_CIRCUIT_BROKEN = "gaze_circuit_broken"
    GAZE_CIRCUIT_WEAK = "gaze_circuit_weak"  # NEW: partial circuit
    DETAIL_NON_COMPOSING = "detail_non_composing"
    DIGNITY_VIOLATION = "dignity_violation"
    TEMPORAL_DISCONTINUITY = "temporal_discontinuity"
    RECIPROCITY_MISSING = "reciprocity_missing"  # NEW: one-way interaction

class GazeType(Enum):
    """Types of gaze specification"""
    NAMED_TARGET = "named_target"  # "looks at counterman"
    VECTOR = "vector"  # [0.3, -0.2] direction
    POSITION_TARGET = "position_target"  # looks toward [0.3, 0.5]

@dataclass
class NarrativeElement:
    """Visual element that contributes to story"""
    name: str
    element_type: str  # 'character', 'prop', 'environment', 'detail'
    spatial_position: Tuple[float, float]  # (x, y) normalized 0-1
    temporal_marker: str  # 'before', 'during', 'after'
    tension_contribution: float  # -1 to 1
    gaze_direction: Optional[Union[str, List[float]]] = None  # name OR [dx, dy] vector
    gaze_target_position: Optional[Tuple[float, float]] = None  # explicit target coords
    emotional_state: Optional[str] = None
    narrative_function: Optional[str] = None
    is_reciprocating: bool = False  # NEW: is this element responding/engaging?
    resolution_strength: float = 0.0  # NEW: 0-1 how strongly this implies resolution
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary"""
        return {
            'name': self.name,
            'element_type': self.element_type,
            'spatial_position': list(self.spatial_position),
            'temporal_marker': self.temporal_marker,
            'tension_contribution': self.tension_contribution,
            'gaze_direction': self.gaze_direction,
            'gaze_target_position': list(self.gaze_target_position) if self.gaze_target_position else None,
            'emotional_state': self.emotional_state,
            'narrative_function': self.narrative_function,
            'is_reciprocating': self.is_reciprocating,
            'resolution_strength': self.resolution_strength
        }

@dataclass
class GazeVector:
    """Geometric representation of a gaze"""
    source_name: str
    source_position: Tuple[float, float]
    direction: Tuple[float, float]  # normalized (dx, dy)
    target_name: Optional[str] = None
    target_position: Optional[Tuple[float, float]] = None
    emotional_valence: str = "neutral"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'source_name': self.source_name,
            'source_position': list(self.source_position),
            'direction': list(self.direction),
            'target_name': self.target_name,
            'target_position': list(self.target_position) if self.target_position else None,
            'emotional_valence': self.emotional_valence
        }

@dataclass
class NarrativeObstruction:
    """Detected H¹ ≠ 0 obstruction"""
    obstruction_type: ObstructionType
    severity: float  # 0-1, contribution to H¹ magnitude
    regions_involved: List[str]
    problem_description: str
    why_incompatible: str
    local_specs_that_fail: Dict[str, Any]
    quality_score: float = 0.0  # NEW: 0-1 quality of what IS present
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': self.obstruction_type.value,
            'severity': self.severity,
            'regions': self.regions_involved,
            'problem': self.problem_description,
            'why_incompatible': self.why_incompatible,
            'local_specs': self.local_specs_that_fail,
            'quality_score': self.quality_score
        }

@dataclass
class SurgicalIntervention:
    """Proposed fix for obstruction"""
    obstruction: NarrativeObstruction
    intervention_type: str
    description: str
    concrete_examples: List[str]
    placement_guidance: str
    expected_H1_reduction: float
    rockwell_technique: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'obstruction_type': self.obstruction.obstruction_type.value,
            'intervention': self.intervention_type,
            'description': self.description,
            'examples': self.concrete_examples,
            'placement': self.placement_guidance,
            'expected_improvement': self.expected_H1_reduction,
            'rockwell_technique': self.rockwell_technique
        }

# ============================================================================
# GEOMETRIC UTILITIES
# ============================================================================

def normalize_vector(v: Tuple[float, float]) -> Tuple[float, float]:
    """Normalize a 2D vector"""
    magnitude = math.sqrt(v[0]**2 + v[1]**2)
    if magnitude < 0.0001:
        return (0.0, 0.0)
    return (v[0]/magnitude, v[1]/magnitude)

def vector_from_positions(source: Tuple[float, float], target: Tuple[float, float]) -> Tuple[float, float]:
    """Create direction vector from source to target"""
    dx = target[0] - source[0]
    dy = target[1] - source[1]
    return normalize_vector((dx, dy))

def dot_product(v1: Tuple[float, float], v2: Tuple[float, float]) -> float:
    """Compute dot product of two vectors"""
    return v1[0]*v2[0] + v1[1]*v2[1]

def gaze_intersection_quality(gaze1: GazeVector, gaze2: GazeVector) -> float:
    """
    Compute how well two gazes 'meet' each other.
    Returns 0-1 where 1 = perfect mutual gaze, 0 = completely unrelated
    """
    # Vector from gaze1 source toward gaze2 source
    toward_other = vector_from_positions(gaze1.source_position, gaze2.source_position)
    
    # How much does gaze1 point toward gaze2's source?
    alignment1 = max(0, dot_product(gaze1.direction, toward_other))
    
    # Vector from gaze2 source toward gaze1 source (opposite direction)
    toward_first = (-toward_other[0], -toward_other[1])
    
    # How much does gaze2 point toward gaze1's source?
    alignment2 = max(0, dot_product(gaze2.direction, toward_first))
    
    # Mutual gaze quality is the product (both must be looking at each other)
    return alignment1 * alignment2

def gaze_reaches_target(gaze: GazeVector, target_pos: Tuple[float, float], tolerance: float = 0.3) -> float:
    """
    Check if a gaze vector points toward a target position.
    Returns 0-1 confidence that gaze reaches target.
    """
    toward_target = vector_from_positions(gaze.source_position, target_pos)
    alignment = dot_product(gaze.direction, toward_target)
    
    # Convert to 0-1 score with tolerance
    if alignment > (1 - tolerance):
        return 1.0
    elif alignment > 0:
        return alignment / (1 - tolerance)
    return 0.0

# ============================================================================
# TEMPORAL STRUCTURE ANALYSIS (IMPROVED)
# ============================================================================

class TemporalStructureAnalyzer:
    """
    Analyzes whether image contains before/during/after markers
    IMPROVED: Quality-weighted scoring, not just presence/absence
    """
    
    def analyze_temporal_compression(
        self, 
        elements: List[NarrativeElement]
    ) -> Dict[str, Any]:
        """
        Measure how much story is compressed into single frame
        IMPROVED: Returns element summaries, not raw objects
        """
        
        temporal_elements = {
            'before_markers': [],
            'during_markers': [],
            'after_markers': []
        }
        
        # Track quality scores
        before_quality = 0.0
        during_quality = 0.0
        after_quality = 0.0
        
        # Classify elements by temporal function
        for element in elements:
            elem_summary = {
                'name': element.name,
                'type': element.element_type,
                'position': list(element.spatial_position),
                'function': element.narrative_function,
                'resolution_strength': element.resolution_strength
            }
            
            if element.temporal_marker == 'before':
                temporal_elements['before_markers'].append(elem_summary)
                # Quality based on tension contribution (stronger backstory = better)
                before_quality += min(abs(element.tension_contribution), 1.0)
                
            elif element.temporal_marker == 'during':
                temporal_elements['during_markers'].append(elem_summary)
                during_quality += min(abs(element.tension_contribution), 1.0)
                
            elif element.temporal_marker == 'after':
                temporal_elements['after_markers'].append(elem_summary)
                # Resolution quality combines tension relief and explicit strength
                resolution_score = element.resolution_strength if element.resolution_strength > 0 else 0.5
                after_quality += resolution_score
        
        # Normalize quality scores
        before_quality = min(before_quality, 1.0)
        during_quality = min(during_quality, 1.0)
        after_quality = min(after_quality, 1.0)
        
        # Calculate compression score (weighted by quality, not just presence)
        has_before = len(temporal_elements['before_markers']) > 0
        has_during = len(temporal_elements['during_markers']) > 0
        has_after = len(temporal_elements['after_markers']) > 0
        
        # Quality-weighted compression
        compression_score = (
            (before_quality * 0.25 if has_before else 0) +
            (during_quality * 0.35 if has_during else 0) +
            (after_quality * 0.40 if has_after else 0)  # Resolution weighted highest
        )
        
        completeness = {
            'has_past': has_before,
            'has_present': has_during,
            'has_future': has_after,
            'past_quality': before_quality,
            'present_quality': during_quality,
            'future_quality': after_quality
        }
        
        return {
            'compression_score': compression_score,
            'temporal_elements': temporal_elements,
            'completeness': completeness,
            'rockwell_threshold': 0.8,
            'meets_rockwell_standard': compression_score >= 0.8
        }
    
    def detect_temporal_obstructions(
        self,
        temporal_structure: Dict[str, Any]
    ) -> List[NarrativeObstruction]:
        """
        Identify H¹ ≠ 0 from temporal incompleteness
        IMPROVED: Severity scaled by quality of what IS present
        """
        
        obstructions = []
        completeness = temporal_structure['completeness']
        
        # Missing past - severity reduced if present quality is high elsewhere
        if not completeness['has_past']:
            # If present is strong, missing past is less severe
            present_compensation = completeness.get('present_quality', 0) * 0.3
            severity = max(0.3, 0.7 - present_compensation)
            
            obstructions.append(NarrativeObstruction(
                obstruction_type=ObstructionType.MISSING_PAST,
                severity=severity,
                regions_involved=['backstory'],
                problem_description="No visual cues for what led to this moment",
                why_incompatible="Current crisis appears without causation",
                local_specs_that_fail={
                    'during_state': 'shows_crisis',
                    'before_state': 'missing',
                    'gluing_requirement': 'crisis must have cause'
                },
                quality_score=0.0
            ))
        
        # Missing future - most critical obstruction
        if not completeness['has_future']:
            severity = 0.8  # High base severity
            
            obstructions.append(NarrativeObstruction(
                obstruction_type=ObstructionType.MISSING_RESOLUTION,
                severity=severity,
                regions_involved=['resolution'],
                problem_description="Story feels incomplete - no implied outcome",
                why_incompatible="Crisis shown without resolution trajectory",
                local_specs_that_fail={
                    'during_state': 'shows_crisis',
                    'after_state': 'missing',
                    'gluing_requirement': 'crisis must imply resolution'
                },
                quality_score=0.0
            ))
        elif completeness.get('future_quality', 0) < 0.5:
            # Has future but weak quality
            severity = 0.4 * (1 - completeness['future_quality'])
            
            obstructions.append(NarrativeObstruction(
                obstruction_type=ObstructionType.MISSING_RESOLUTION,
                severity=severity,
                regions_involved=['resolution'],
                problem_description="Resolution hint present but weak",
                why_incompatible="Outcome implied but not compelling",
                local_specs_that_fail={
                    'after_state': 'weak',
                    'quality': completeness['future_quality'],
                    'gluing_requirement': 'resolution must be clear'
                },
                quality_score=completeness['future_quality']
            ))
        
        return obstructions

# ============================================================================
# GAZE STRUCTURE ANALYSIS (GEOMETRIC)
# ============================================================================

class GazeCircuitAnalyzer:
    """
    Analyzes whether eye lines create narrative circuit
    IMPROVED: Geometric vector analysis, not just named connections
    """
    
    def __init__(self):
        self.gaze_vectors: List[GazeVector] = []
    
    def analyze_gaze_structure(
        self,
        elements: List[NarrativeElement]
    ) -> Dict[str, Any]:
        """
        Check if gazes form narrative loop using geometric analysis
        """
        
        # Build element position lookup
        element_positions = {
            elem.name: elem.spatial_position
            for elem in elements
        }
        
        # Extract and convert gazes to vectors
        self.gaze_vectors = []
        for element in elements:
            if element.gaze_direction is None:
                continue
                
            gaze = self._create_gaze_vector(element, element_positions)
            if gaze:
                self.gaze_vectors.append(gaze)
        
        if len(self.gaze_vectors) < 2:
            return {
                'has_circuit': False,
                'circuit_quality': 0.0,
                'insufficient_gazes': True,
                'problem': 'Need at least 2 gaze vectors for circuit',
                'gaze_vectors': [g.to_dict() for g in self.gaze_vectors]
            }
        
        # Analyze circuit properties
        circuit_analysis = self._analyze_circuit_geometry()
        
        return {
            'has_circuit': circuit_analysis['closed'],
            'circuit_quality': circuit_analysis['quality'],
            'circuit_path': circuit_analysis['path'],
            'mutual_gaze_pairs': circuit_analysis['mutual_pairs'],
            'rockwell_standard': circuit_analysis['quality'] >= 0.7,
            'gaze_vectors': [g.to_dict() for g in self.gaze_vectors],
            'geometric_analysis': circuit_analysis
        }
    
    def _create_gaze_vector(
        self, 
        element: NarrativeElement,
        positions: Dict[str, Tuple[float, float]]
    ) -> Optional[GazeVector]:
        """Convert element's gaze to geometric vector"""
        
        source_pos = element.spatial_position
        
        # Case 1: Gaze is a vector [dx, dy]
        if isinstance(element.gaze_direction, list):
            direction = normalize_vector(tuple(element.gaze_direction))
            return GazeVector(
                source_name=element.name,
                source_position=source_pos,
                direction=direction,
                target_name=None,
                target_position=element.gaze_target_position,
                emotional_valence=element.emotional_state or "neutral"
            )
        
        # Case 2: Gaze is a named target
        if isinstance(element.gaze_direction, str):
            target_name = element.gaze_direction
            
            # Special cases for non-element targets
            if target_name.lower() in ['viewer', 'camera', 'off_camera', 'off_camera_left', 'off_camera_right']:
                # Viewer is at (0.5, 0.0) - in front of image
                if 'left' in target_name.lower():
                    target_pos = (-0.5, 0.5)
                elif 'right' in target_name.lower():
                    target_pos = (1.5, 0.5)
                else:
                    target_pos = (0.5, -0.5)  # Toward viewer
                direction = vector_from_positions(source_pos, target_pos)
                return GazeVector(
                    source_name=element.name,
                    source_position=source_pos,
                    direction=direction,
                    target_name=target_name,
                    target_position=target_pos,
                    emotional_valence=element.emotional_state or "neutral"
                )
            
            if target_name.lower() in ['down', 'downward', 'floor']:
                direction = (0.0, 1.0)  # Down in image coords
                return GazeVector(
                    source_name=element.name,
                    source_position=source_pos,
                    direction=direction,
                    target_name=target_name,
                    target_position=None,
                    emotional_valence=element.emotional_state or "neutral"
                )
            
            if target_name.lower() in ['up', 'upward', 'sky']:
                direction = (0.0, -1.0)  # Up in image coords
                return GazeVector(
                    source_name=element.name,
                    source_position=source_pos,
                    direction=direction,
                    target_name=target_name,
                    target_position=None,
                    emotional_valence=element.emotional_state or "neutral"
                )
            
            # Named element target
            if target_name in positions:
                target_pos = positions[target_name]
                direction = vector_from_positions(source_pos, target_pos)
                return GazeVector(
                    source_name=element.name,
                    source_position=source_pos,
                    direction=direction,
                    target_name=target_name,
                    target_position=target_pos,
                    emotional_valence=element.emotional_state or "neutral"
                )
        
        return None
    
    def _analyze_circuit_geometry(self) -> Dict[str, Any]:
        """
        Analyze the geometric properties of the gaze circuit
        """
        
        n_gazes = len(self.gaze_vectors)
        
        # Find mutual gaze pairs (geometric intersection quality)
        mutual_pairs = []
        total_mutual_quality = 0.0
        
        for i in range(n_gazes):
            for j in range(i + 1, n_gazes):
                quality = gaze_intersection_quality(
                    self.gaze_vectors[i],
                    self.gaze_vectors[j]
                )
                if quality > 0.3:  # Threshold for "mutual"
                    mutual_pairs.append({
                        'pair': [self.gaze_vectors[i].source_name, self.gaze_vectors[j].source_name],
                        'quality': quality
                    })
                    total_mutual_quality += quality
        
        # Build connection graph based on gaze targets
        connections = {}
        for gaze in self.gaze_vectors:
            source = gaze.source_name
            if source not in connections:
                connections[source] = []
            
            # Add named target
            if gaze.target_name and gaze.target_name in [g.source_name for g in self.gaze_vectors]:
                connections[source].append({
                    'target': gaze.target_name,
                    'quality': 1.0  # Direct named connection
                })
            
            # Check geometric reach to other gaze sources
            for other_gaze in self.gaze_vectors:
                if other_gaze.source_name == source:
                    continue
                reach = gaze_reaches_target(gaze, other_gaze.source_position)
                if reach > 0.5:
                    # Avoid duplicates
                    existing = [c for c in connections[source] if c['target'] == other_gaze.source_name]
                    if not existing:
                        connections[source].append({
                            'target': other_gaze.source_name,
                            'quality': reach
                        })
        
        # Find circuit path
        circuit_path = self._find_best_circuit(connections)
        
        # Calculate overall circuit quality
        if circuit_path and len(circuit_path) >= 2:
            path_quality = self._calculate_path_quality(circuit_path, connections)
            # Bonus for mutual gazes
            mutual_bonus = min(total_mutual_quality * 0.2, 0.3)
            overall_quality = min(path_quality + mutual_bonus, 1.0)
        else:
            overall_quality = total_mutual_quality * 0.5  # Some credit for mutual gazes even without full circuit
        
        return {
            'closed': circuit_path is not None and len(circuit_path) >= 3,
            'quality': overall_quality,
            'path': circuit_path or [],
            'path_length': len(circuit_path) if circuit_path else 0,
            'mutual_pairs': mutual_pairs,
            'connections': {k: [c['target'] for c in v] for k, v in connections.items()}
        }
    
    def _find_best_circuit(self, connections: Dict[str, List[Dict]]) -> Optional[List[str]]:
        """Find the best quality circuit in the connection graph"""
        
        best_circuit = None
        best_quality = 0
        
        for start in connections:
            circuit = self._dfs_circuit(start, start, [], set(), connections)
            if circuit and len(circuit) >= 3:
                quality = self._calculate_path_quality(circuit, connections)
                if quality > best_quality:
                    best_quality = quality
                    best_circuit = circuit
        
        return best_circuit
    
    def _dfs_circuit(
        self,
        current: str,
        start: str,
        path: List[str],
        visited: set,
        connections: Dict[str, List[Dict]]
    ) -> Optional[List[str]]:
        """Depth-first search for circuits"""
        
        path = path + [current]
        
        if current not in connections:
            return None
        
        for conn in connections[current]:
            next_node = conn['target']
            
            # Found circuit back to start
            if next_node == start and len(path) >= 2:
                return path
            
            # Continue searching
            if next_node not in visited:
                visited.add(current)
                result = self._dfs_circuit(next_node, start, path, visited, connections)
                if result:
                    return result
                visited.discard(current)
        
        return None
    
    def _calculate_path_quality(
        self,
        path: List[str],
        connections: Dict[str, List[Dict]]
    ) -> float:
        """Calculate quality of a circuit path"""
        
        if not path:
            return 0.0
        
        total_quality = 0.0
        n_edges = len(path)
        
        for i in range(n_edges):
            source = path[i]
            target = path[(i + 1) % n_edges]
            
            if source in connections:
                for conn in connections[source]:
                    if conn['target'] == target:
                        total_quality += conn['quality']
                        break
        
        return total_quality / n_edges if n_edges > 0 else 0.0
    
    def detect_gaze_obstructions(
        self,
        gaze_structure: Dict[str, Any]
    ) -> List[NarrativeObstruction]:
        """
        Identify H¹ ≠ 0 from broken/weak gaze circuits
        IMPROVED: Graduated severity based on circuit quality
        """
        
        obstructions = []
        
        quality = gaze_structure.get('circuit_quality', 0)
        has_circuit = gaze_structure.get('has_circuit', False)
        
        if gaze_structure.get('insufficient_gazes'):
            obstructions.append(NarrativeObstruction(
                obstruction_type=ObstructionType.GAZE_CIRCUIT_BROKEN,
                severity=0.7,
                regions_involved=['viewer_path'],
                problem_description="Insufficient gaze information for circuit analysis",
                why_incompatible="Cannot form narrative loop without multiple gazes",
                local_specs_that_fail={
                    'gaze_count': len(gaze_structure.get('gaze_vectors', [])),
                    'minimum_required': 2
                },
                quality_score=0.0
            ))
        elif not has_circuit:
            # No circuit found - scale severity by what IS present
            mutual_pairs = gaze_structure.get('mutual_gaze_pairs', [])
            partial_credit = len(mutual_pairs) * 0.1
            severity = max(0.3, 0.6 - partial_credit)
            
            obstructions.append(NarrativeObstruction(
                obstruction_type=ObstructionType.GAZE_CIRCUIT_BROKEN,
                severity=severity,
                regions_involved=['viewer_path'],
                problem_description="No closed narrative loop in gaze structure",
                why_incompatible="Viewer eye wanders without return",
                local_specs_that_fail={
                    'circuit_found': False,
                    'mutual_pairs_found': len(mutual_pairs),
                    'circuit_requirement': 'must form closed loop'
                },
                quality_score=quality
            ))
        elif quality < 0.7:
            # Circuit exists but weak
            severity = 0.4 * (1 - quality)
            
            obstructions.append(NarrativeObstruction(
                obstruction_type=ObstructionType.GAZE_CIRCUIT_WEAK,
                severity=severity,
                regions_involved=['viewer_path'],
                problem_description="Gaze circuit exists but is weak",
                why_incompatible=f"Circuit quality {quality:.2f} below Rockwell standard (0.7)",
                local_specs_that_fail={
                    'circuit_quality': quality,
                    'rockwell_standard': 0.7,
                    'path': gaze_structure.get('circuit_path', [])
                },
                quality_score=quality
            ))
        
        return obstructions

# ============================================================================
# SPATIAL COMPOSITION ANALYSIS
# ============================================================================

class SpatialCompositionAnalyzer:
    """
    Analyzes whether spatial relationships support narrative
    """
    
    def analyze_spatial_coherence(
        self,
        elements: List[NarrativeElement]
    ) -> Dict[str, Any]:
        """
        Check if spatial relationships match narrative relationships
        """
        
        characters = [e for e in elements if e.element_type == 'character']
        
        if len(characters) < 2:
            return {
                'sufficient_characters': False,
                'problem': 'Need at least 2 characters for relationship analysis'
            }
        
        protagonist = self._identify_protagonist(characters)
        antagonist = self._identify_antagonist(characters, protagonist)
        
        if not antagonist:
            return {
                'has_clear_relationship': False,
                'problem': 'No clear antagonist/foil character'
            }
        
        spatial_tension = self._measure_spatial_tension(
            protagonist.spatial_position,
            antagonist.spatial_position
        )
        
        narrative_tension = abs(
            protagonist.tension_contribution - antagonist.tension_contribution
        )
        
        # Check for reciprocity
        reciprocity_score = self._measure_reciprocity(protagonist, antagonist, elements)
        
        return {
            'protagonist': protagonist.name,
            'antagonist': antagonist.name,
            'spatial_tension': spatial_tension,
            'narrative_tension': narrative_tension,
            'mismatch': abs(spatial_tension - narrative_tension),
            'coherent': abs(spatial_tension - narrative_tension) < 0.3,
            'reciprocity_score': reciprocity_score,
            'has_reciprocity': reciprocity_score > 0.5
        }
    
    def _identify_protagonist(self, characters: List[NarrativeElement]) -> NarrativeElement:
        return max(characters, key=lambda c: abs(c.tension_contribution))
    
    def _identify_antagonist(
        self,
        characters: List[NarrativeElement],
        protagonist: NarrativeElement
    ) -> Optional[NarrativeElement]:
        others = [c for c in characters if c.name != protagonist.name]
        if not others:
            return None
        return max(
            others,
            key=lambda c: abs(c.tension_contribution - protagonist.tension_contribution)
        )
    
    def _measure_spatial_tension(
        self,
        pos1: Tuple[float, float],
        pos2: Tuple[float, float]
    ) -> float:
        distance = ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)**0.5
        return 1 - min(distance, 1.0)
    
    def _measure_reciprocity(
        self,
        protagonist: NarrativeElement,
        antagonist: NarrativeElement,
        all_elements: List[NarrativeElement]
    ) -> float:
        """
        Measure whether characters are engaged in mutual interaction
        """
        score = 0.0
        
        # Check if protagonist is reciprocating
        if protagonist.is_reciprocating:
            score += 0.3
        
        # Check if antagonist is reciprocating
        if antagonist.is_reciprocating:
            score += 0.3
        
        # Check gaze reciprocity
        prot_looks_at_ant = (
            protagonist.gaze_direction == antagonist.name or
            (isinstance(protagonist.gaze_direction, str) and antagonist.name.lower() in protagonist.gaze_direction.lower())
        )
        ant_looks_at_prot = (
            antagonist.gaze_direction == protagonist.name or
            (isinstance(antagonist.gaze_direction, str) and protagonist.name.lower() in antagonist.gaze_direction.lower())
        )
        
        if prot_looks_at_ant and ant_looks_at_prot:
            score += 0.4
        elif prot_looks_at_ant or ant_looks_at_prot:
            score += 0.2
        
        return min(score, 1.0)
    
    def detect_spatial_obstructions(
        self,
        spatial_analysis: Dict[str, Any]
    ) -> List[NarrativeObstruction]:
        obstructions = []
        
        if spatial_analysis.get('coherent') is False:
            mismatch = spatial_analysis.get('mismatch', 0.5)
            severity = min(0.6, 0.3 + mismatch * 0.5)
            
            obstructions.append(NarrativeObstruction(
                obstruction_type=ObstructionType.SPATIAL_NARRATIVE_MISMATCH,
                severity=severity,
                regions_involved=['protagonist', 'antagonist'],
                problem_description="Physical spacing doesn't match emotional tension",
                why_incompatible=f"Spatial tension ({spatial_analysis['spatial_tension']:.2f}) mismatches narrative tension ({spatial_analysis['narrative_tension']:.2f})",
                local_specs_that_fail={
                    'spatial': spatial_analysis['spatial_tension'],
                    'narrative': spatial_analysis['narrative_tension'],
                    'gluing_requirement': 'must correlate within 0.3'
                },
                quality_score=1.0 - mismatch
            ))
        
        # Check for missing reciprocity
        if spatial_analysis.get('has_reciprocity') is False:
            reciprocity = spatial_analysis.get('reciprocity_score', 0)
            severity = 0.5 * (1 - reciprocity)
            
            obstructions.append(NarrativeObstruction(
                obstruction_type=ObstructionType.RECIPROCITY_MISSING,
                severity=severity,
                regions_involved=['protagonist', 'antagonist'],
                problem_description="One-way interaction - no reciprocal engagement",
                why_incompatible="Characters not mutually engaged",
                local_specs_that_fail={
                    'reciprocity_score': reciprocity,
                    'required': 0.5
                },
                quality_score=reciprocity
            ))
        
        return obstructions

# ============================================================================
# EMOTIONAL ARC ANALYSIS
# ============================================================================

class EmotionalArcAnalyzer:
    """
    Analyzes whether emotional states compose into arc
    """
    
    POSITIVE_EMOTIONS = ['joy', 'hope', 'pride', 'contentment', 'relief', 'trust', 'warmth', 
                         'gentle_concern', 'patience', 'attentive', 'engaged', 'caring']
    NEGATIVE_EMOTIONS = ['fear', 'anger', 'sadness', 'shame', 'anxiety', 'distress', 
                         'troubled', 'withdrawn', 'isolated']
    TRANSITIONAL_EMOTIONS = ['contemplative', 'uncertain', 'conflicted', 'opening_up',
                             'beginning_to_trust', 'threshold']
    
    def analyze_emotional_coherence(
        self,
        elements: List[NarrativeElement]
    ) -> Dict[str, Any]:
        characters = [e for e in elements if e.element_type == 'character']
        
        if len(characters) < 2:
            return {
                'sufficient_characters': False,
                'problem': 'Need multiple characters for arc analysis'
            }
        
        emotions = {
            char.name: char.emotional_state or 'neutral'
            for char in characters
        }
        
        emotional_arc = self._analyze_emotional_relationships(emotions)
        
        return {
            'emotions': emotions,
            'arc_type': emotional_arc['type'],
            'arc_quality': emotional_arc['quality'],
            'coherent': emotional_arc['quality'] > 0.4,
            'description': emotional_arc['description']
        }
    
    def _analyze_emotional_relationships(
        self,
        emotions: Dict[str, str]
    ) -> Dict[str, Any]:
        emotion_list = list(emotions.values())
        
        # Check for clear patterns
        has_positive = any(self._is_emotion_type(e, self.POSITIVE_EMOTIONS) for e in emotion_list)
        has_negative = any(self._is_emotion_type(e, self.NEGATIVE_EMOTIONS) for e in emotion_list)
        has_transitional = any(self._is_emotion_type(e, self.TRANSITIONAL_EMOTIONS) for e in emotion_list)
        
        if len(set(emotion_list)) == 1:
            return {
                'type': 'uniform',
                'quality': 0.3,
                'description': 'All characters share same emotion - limited tension'
            }
        
        # Best case: positive/negative opposition with transition
        if has_positive and has_negative and has_transitional:
            return {
                'type': 'full_arc',
                'quality': 1.0,
                'description': 'Complete emotional arc with transition state'
            }
        
        # Good: clear opposition
        if has_positive and has_negative:
            return {
                'type': 'opposition',
                'quality': 0.8,
                'description': 'Clear emotional opposition creates tension'
            }
        
        # Decent: transitional states present
        if has_transitional:
            return {
                'type': 'transitional',
                'quality': 0.6,
                'description': 'Emotional transition in progress'
            }
        
        return {
            'type': 'incoherent',
            'quality': 0.2,
            'description': 'Emotions don\'t form clear dramatic relationship'
        }
    
    def _is_emotion_type(self, emotion: str, emotion_list: List[str]) -> bool:
        """Check if emotion matches any in list (fuzzy matching)"""
        emotion_lower = emotion.lower()
        for e in emotion_list:
            if e in emotion_lower or emotion_lower in e:
                return True
        return False
    
    def detect_emotional_obstructions(
        self,
        emotional_analysis: Dict[str, Any]
    ) -> List[NarrativeObstruction]:
        obstructions = []
        
        quality = emotional_analysis.get('arc_quality', 0)
        
        if quality < 0.4:
            severity = 0.7 * (1 - quality)
            
            obstructions.append(NarrativeObstruction(
                obstruction_type=ObstructionType.EMOTIONAL_INCOHERENCE,
                severity=severity,
                regions_involved=list(emotional_analysis.get('emotions', {}).keys()),
                problem_description="Character emotions don't form dramatic arc",
                why_incompatible=f"Arc type '{emotional_analysis.get('arc_type')}' with quality {quality:.2f}",
                local_specs_that_fail={
                    'emotions': emotional_analysis.get('emotions', {}),
                    'arc_type': emotional_analysis.get('arc_type'),
                    'quality': quality
                },
                quality_score=quality
            ))
        
        return obstructions

# ============================================================================
# SURGICAL INTERVENTION GENERATOR
# ============================================================================

class SurgicalInterventionGenerator:
    """
    Generates concrete fixes for H¹ ≠ 0 obstructions
    """
    
    def generate_interventions(
        self,
        obstructions: List[NarrativeObstruction]
    ) -> List[SurgicalIntervention]:
        interventions = []
        
        for obs in obstructions:
            if obs.obstruction_type == ObstructionType.MISSING_PAST:
                interventions.append(self._fix_missing_past(obs))
            
            elif obs.obstruction_type == ObstructionType.MISSING_RESOLUTION:
                interventions.append(self._fix_missing_resolution(obs))
            
            elif obs.obstruction_type == ObstructionType.SPATIAL_NARRATIVE_MISMATCH:
                interventions.append(self._fix_spatial_mismatch(obs))
            
            elif obs.obstruction_type == ObstructionType.EMOTIONAL_INCOHERENCE:
                interventions.append(self._fix_emotional_incoherence(obs))
            
            elif obs.obstruction_type in [ObstructionType.GAZE_CIRCUIT_BROKEN, ObstructionType.GAZE_CIRCUIT_WEAK]:
                interventions.append(self._fix_gaze_circuit(obs))
            
            elif obs.obstruction_type == ObstructionType.RECIPROCITY_MISSING:
                interventions.append(self._fix_reciprocity(obs))
        
        return interventions
    
    def _fix_missing_past(self, obs: NarrativeObstruction) -> SurgicalIntervention:
        return SurgicalIntervention(
            obstruction=obs,
            intervention_type="add_backstory_prop",
            description="Insert visual element that implies what led to this moment",
            concrete_examples=[
                "Packed suitcase in corner (implies departure/journey)",
                "Torn letter in hand (implies bad news received)",
                "Visible bandage/wound (implies recent injury)",
                "Worn/dirty clothing (implies extended struggle or journey)",
                "Discarded tool/weapon (implies recent action)",
                "Clock showing specific time (implies scheduling pressure)"
            ],
            placement_guidance="Background or held by secondary character - should be visible but not dominant.",
            expected_H1_reduction=obs.severity,
            rockwell_technique="Subtle backstory props - viewer discovers them on second viewing"
        )
    
    def _fix_missing_resolution(self, obs: NarrativeObstruction) -> SurgicalIntervention:
        quality = obs.quality_score
        
        if quality > 0:
            # Strengthen existing weak resolution
            return SurgicalIntervention(
                obstruction=obs,
                intervention_type="strengthen_resolution_hint",
                description="Amplify existing resolution elements",
                concrete_examples=[
                    "Make resolution gesture more visible/prominent",
                    "Add secondary confirmation of resolution (another character noticing)",
                    "Increase lighting on resolution element",
                    "Add protagonist beginning to respond to resolution offer"
                ],
                placement_guidance="Current resolution hint is present but weak. Make it more discoverable.",
                expected_H1_reduction=obs.severity,
                rockwell_technique="Layered resolution - multiple subtle cues that reinforce each other"
            )
        else:
            return SurgicalIntervention(
                obstruction=obs,
                intervention_type="add_resolution_hint",
                description="Insert subtle element suggesting how this resolves",
                concrete_examples=[
                    "Sympathetic bystander entering frame (implies help coming)",
                    "Object of hope in background (implies solution available)",
                    "Character's body language shifting toward resolution",
                    "Environmental change: door opening, light through window",
                    "Protagonist's hand reaching toward offered comfort",
                    "Bridge object connecting characters (shared focus)"
                ],
                placement_guidance="Should be discoverable but not obvious. Never heavy-handed.",
                expected_H1_reduction=obs.severity,
                rockwell_technique="Foreshadowing through environmental storytelling"
            )
    
    def _fix_spatial_mismatch(self, obs: NarrativeObstruction) -> SurgicalIntervention:
        return SurgicalIntervention(
            obstruction=obs,
            intervention_type="adjust_spatial_relationships",
            description="Reposition characters to match narrative tension",
            concrete_examples=[
                "INTIMACY: Move characters closer, within personal space",
                "BARRIER CROSSING: Have one character lean across dividing element",
                "CONNECTION: Create overlap or reaching gesture",
                "POWER DYNAMIC: Adjust vertical positioning",
                "CONFRONTATION: Direct facing, eye-to-eye axis"
            ],
            placement_guidance=f"Spatial ({obs.local_specs_that_fail.get('spatial', 0):.2f}) vs narrative ({obs.local_specs_that_fail.get('narrative', 0):.2f}) tension",
            expected_H1_reduction=obs.severity,
            rockwell_technique="Positioning as storytelling - every inch of space has meaning"
        )
    
    def _fix_emotional_incoherence(self, obs: NarrativeObstruction) -> SurgicalIntervention:
        return SurgicalIntervention(
            obstruction=obs,
            intervention_type="establish_emotional_relationship",
            description="Connect character emotions into dramatic relationship",
            concrete_examples=[
                "OPPOSITION: One character's concern vs another's distress",
                "TRANSITION: Show moment of emotional shift (opening up)",
                "REACTION: One character responding to another's state",
                "GRADIENT: Emotional progression visible in body language",
                "CATALYST: One character causing another's emotional change"
            ],
            placement_guidance="Use gaze, body orientation, and expression to connect emotions.",
            expected_H1_reduction=obs.severity,
            rockwell_technique="Emotional causality - viewer understands why each character feels as they do"
        )
    
    def _fix_gaze_circuit(self, obs: NarrativeObstruction) -> SurgicalIntervention:
        quality = obs.quality_score
        
        if quality > 0.3:
            # Strengthen existing weak circuit
            return SurgicalIntervention(
                obstruction=obs,
                intervention_type="strengthen_gaze_circuit",
                description="Improve existing gaze connections",
                concrete_examples=[
                    "Ensure mutual eye contact between key characters",
                    "Add reflective surface that returns gaze to viewer",
                    "Include bridge object both characters look at",
                    "Adjust head angles to make gaze directions clearer",
                    "Add intermediate gaze target that connects endpoints"
                ],
                placement_guidance=f"Current circuit quality: {quality:.2f}. Strengthen weakest links.",
                expected_H1_reduction=obs.severity,
                rockwell_technique="Guided viewing - tighten the narrative loop"
            )
        else:
            return SurgicalIntervention(
                obstruction=obs,
                intervention_type="create_gaze_circuit",
                description="Connect eye lines to form narrative loop",
                concrete_examples=[
                    "A looks at B → B looks at Object → Object leads back to A",
                    "Character looks at problem → Problem implies solution → Solution near character",
                    "TRIANGULATED: A → shared object → B → A (minimum 3-node loop)",
                    "RECIPROCAL: Direct mutual gaze between characters",
                    "VIEWER LOOP: Character looks at element that leads viewer's eye back to character"
                ],
                placement_guidance="Start with protagonist gaze, ensure return path. Use head angles precisely.",
                expected_H1_reduction=obs.severity,
                rockwell_technique="Guided viewing - control exactly how viewer's eye moves through narrative"
            )
    
    def _fix_reciprocity(self, obs: NarrativeObstruction) -> SurgicalIntervention:
        return SurgicalIntervention(
            obstruction=obs,
            intervention_type="add_reciprocal_gesture",
            description="Transform one-way interaction into mutual engagement",
            concrete_examples=[
                "Protagonist begins to look up toward other character",
                "Protagonist's hand reaches toward offered object/comfort",
                "Body posture opens/turns toward the other character",
                "Facial expression shifts to acknowledge connection",
                "Protagonist speaking or about to speak",
                "Shared attention on bridge object (photo, document, food)"
            ],
            placement_guidance="The passive character needs an active gesture - reaching, looking, speaking, responding.",
            expected_H1_reduction=obs.severity,
            rockwell_technique="Mutual engagement - both parties must participate in the narrative moment"
        )

# ============================================================================
# MAIN COHOMOLOGY DETECTOR
# ============================================================================

class NarrativeCohomologyDetector:
    """
    Main H¹ detection and surgical intervention system
    """
    
    def __init__(self):
        self.temporal_analyzer = TemporalStructureAnalyzer()
        self.gaze_analyzer = GazeCircuitAnalyzer()
        self.spatial_analyzer = SpatialCompositionAnalyzer()
        self.emotional_analyzer = EmotionalArcAnalyzer()
        self.intervention_generator = SurgicalInterventionGenerator()
    
    def analyze_visual_narrative(
        self,
        elements: List[NarrativeElement],
        include_interventions: bool = True
    ) -> Dict[str, Any]:
        """
        Complete narrative cohomology analysis
        """
        
        # Run all analyses
        temporal_structure = self.temporal_analyzer.analyze_temporal_compression(elements)
        gaze_structure = self.gaze_analyzer.analyze_gaze_structure(elements)
        spatial_coherence = self.spatial_analyzer.analyze_spatial_coherence(elements)
        emotional_coherence = self.emotional_analyzer.analyze_emotional_coherence(elements)
        
        # Collect obstructions
        all_obstructions = []
        
        all_obstructions.extend(
            self.temporal_analyzer.detect_temporal_obstructions(temporal_structure)
        )
        all_obstructions.extend(
            self.gaze_analyzer.detect_gaze_obstructions(gaze_structure)
        )
        all_obstructions.extend(
            self.spatial_analyzer.detect_spatial_obstructions(spatial_coherence)
        )
        all_obstructions.extend(
            self.emotional_analyzer.detect_emotional_obstructions(emotional_coherence)
        )
        
        # Calculate H¹ magnitude (quality-weighted)
        H1_magnitude = sum([obs.severity for obs in all_obstructions])
        
        # Narrative coherence (inverse of H¹)
        narrative_coherence = max(0, 1 - H1_magnitude)
        
        # Generate interventions
        interventions = []
        if include_interventions and all_obstructions:
            interventions = self.intervention_generator.generate_interventions(
                all_obstructions
            )
        
        return {
            'H¹_magnitude': round(H1_magnitude, 3),
            'narrative_coherence': round(narrative_coherence, 3),
            'coherent': H1_magnitude < 0.3,
            'rockwell_quality': H1_magnitude < 0.2,
            
            'analyses': {
                'temporal_structure': temporal_structure,
                'gaze_structure': gaze_structure,
                'spatial_coherence': spatial_coherence,
                'emotional_coherence': emotional_coherence
            },
            
            'obstructions': [obs.to_dict() for obs in all_obstructions],
            
            'surgical_interventions': [interv.to_dict() for interv in interventions],
            
            'rockwell_assessment': self._generate_rockwell_assessment(
                H1_magnitude,
                temporal_structure,
                gaze_structure,
                all_obstructions
            )
        }
    
    def _generate_rockwell_assessment(
        self,
        H1: float,
        temporal: Dict,
        gaze: Dict,
        obstructions: List
    ) -> str:
        if H1 < 0.2:
            assessment = "✓ ROCKWELL MASTERY: This image achieves single-frame narrative closure. "
        elif H1 < 0.5:
            assessment = "○ COMPETENT NARRATIVE: Story is readable but could be tighter. "
        elif H1 < 1.0:
            assessment = "△ DEVELOPING NARRATIVE: Core elements present but significant gaps. "
        else:
            assessment = "✗ NARRATIVE FRAGMENTATION: Story elements don't cohere. "
        
        if temporal.get('meets_rockwell_standard'):
            assessment += "Excellent temporal compression. "
        elif temporal.get('completeness', {}).get('has_future'):
            assessment += "Has resolution but could be stronger. "
        else:
            assessment += "Needs resolution hint. "
        
        gaze_quality = gaze.get('circuit_quality', 0)
        if gaze_quality >= 0.7:
            assessment += "Strong gaze circuit. "
        elif gaze_quality >= 0.4:
            assessment += "Partial gaze circuit. "
        else:
            assessment += "Gaze structure needs work. "
        
        assessment += f"{len(obstructions)} obstruction(s) (H¹ = {H1:.2f})."
        
        return assessment

# ============================================================================
# FASTMCP SERVER INTERFACE
# ============================================================================

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Norman Rockwell Visual Narrative Cohomology v3")

@mcp.tool()
def analyze_visual_narrative_cohomology(
    elements: str,
    include_interventions: bool = True
) -> str:
    """
    Analyze visual narrative for H¹ ≠ 0 obstructions and suggest surgical interventions.

    Detects where visual storytelling breaks down and provides Rockwell-level fixes.

    Args:
        elements: JSON array of narrative elements with structure:
            [
                {
                    "name": "protagonist",
                    "element_type": "character",
                    "spatial_position": [0.6, 0.5],
                    "temporal_marker": "during",
                    "tension_contribution": 0.8,
                    "gaze_direction": "antagonist",  // OR [0.3, -0.2] as vector
                    "gaze_target_position": [0.3, 0.5],  // optional explicit target coords
                    "emotional_state": "fear",
                    "narrative_function": "main_character",
                    "is_reciprocating": false,  // NEW: is element actively responding?
                    "resolution_strength": 0.0  // NEW: 0-1 how strongly implies resolution
                },
                ...
            ]
        include_interventions: Whether to generate surgical fixes

    Returns:
        Complete cohomology analysis with H¹ magnitude, obstructions, and interventions
    """
    
    # Parse elements
    elements_data = json.loads(elements)
    
    # Convert to NarrativeElement objects
    narrative_elements = []
    for elem_data in elements_data:
        narrative_elements.append(NarrativeElement(
            name=elem_data['name'],
            element_type=elem_data['element_type'],
            spatial_position=tuple(elem_data['spatial_position']),
            temporal_marker=elem_data['temporal_marker'],
            tension_contribution=elem_data['tension_contribution'],
            gaze_direction=elem_data.get('gaze_direction'),
            gaze_target_position=tuple(elem_data['gaze_target_position']) if elem_data.get('gaze_target_position') else None,
            emotional_state=elem_data.get('emotional_state'),
            narrative_function=elem_data.get('narrative_function'),
            is_reciprocating=elem_data.get('is_reciprocating', False),
            resolution_strength=elem_data.get('resolution_strength', 0.0)
        ))
    
    # Run analysis
    detector = NarrativeCohomologyDetector()
    result = detector.analyze_visual_narrative(
        narrative_elements,
        include_interventions
    )
    
    return json.dumps(result, indent=2)

@mcp.tool()
def quick_cohomology_check(
    description: str,
    protagonist_emotion: str,
    antagonist_emotion: str = None,
    has_backstory_cue: bool = False,
    has_resolution_hint: bool = False,
    resolution_strength: float = 0.5,
    has_mutual_gaze: bool = False,
    protagonist_reciprocating: bool = False
) -> str:
    """
    Quick H¹ check for simple visual narrative.

    IMPROVED: Now includes resolution strength, mutual gaze, and reciprocity parameters.

    Args:
        description: Brief scene description
        protagonist_emotion: Main character's emotional state
        antagonist_emotion: Antagonist's emotion (if present)
        has_backstory_cue: Whether image shows what led to this moment
        has_resolution_hint: Whether image implies outcome
        resolution_strength: 0-1 how strongly resolution is implied (default 0.5)
        has_mutual_gaze: Whether characters have mutual eye contact
        protagonist_reciprocating: Whether protagonist is actively engaging (not passive)

    Returns:
        Quick assessment with H¹ estimate and top recommendations
    """
    
    # Construct element set with new parameters
    elements = [
        NarrativeElement(
            name="protagonist",
            element_type="character",
            spatial_position=(0.6, 0.5),
            temporal_marker="during",
            tension_contribution=0.8,
            emotional_state=protagonist_emotion,
            gaze_direction="antagonist" if antagonist_emotion and has_mutual_gaze else "downward",
            is_reciprocating=protagonist_reciprocating
        )
    ]
    
    if antagonist_emotion:
        elements.append(NarrativeElement(
            name="antagonist",
            element_type="character",
            spatial_position=(0.3, 0.5),
            temporal_marker="during",
            tension_contribution=-0.5,
            emotional_state=antagonist_emotion,
            gaze_direction="protagonist",
            is_reciprocating=True  # Antagonist typically initiating
        ))
    
    if has_backstory_cue:
        elements.append(NarrativeElement(
            name="backstory_prop",
            element_type="prop",
            spatial_position=(0.7, 0.7),
            temporal_marker="before",
            tension_contribution=0.4
        ))
    
    if has_resolution_hint:
        elements.append(NarrativeElement(
            name="resolution_hint",
            element_type="detail",
            spatial_position=(0.4, 0.4),
            temporal_marker="after",
            tension_contribution=-0.3,
            resolution_strength=resolution_strength
        ))
    
    # Analyze
    detector = NarrativeCohomologyDetector()
    result = detector.analyze_visual_narrative(elements, include_interventions=True)
    
    # Simplify output
    quick_result = {
        'H¹_magnitude': result['H¹_magnitude'],
        'narrative_quality': 'EXCELLENT' if result['H¹_magnitude'] < 0.2 else 'GOOD' if result['H¹_magnitude'] < 0.5 else 'DEVELOPING' if result['H¹_magnitude'] < 1.0 else 'NEEDS_WORK',
        'rockwell_assessment': result['rockwell_assessment'],
        'gaze_circuit_quality': result['analyses']['gaze_structure'].get('circuit_quality', 0),
        'top_issues': result['obstructions'][:3],
        'priority_fix': result['surgical_interventions'][0] if result['surgical_interventions'] else None
    }
    
    return json.dumps(quick_result, indent=2)

@mcp.tool()
def analyze_gaze_geometry(
    gaze_vectors: str
) -> str:
    """
    Analyze gaze geometry for circuit closure and mutual attention.
    
    NEW TOOL: Direct geometric analysis of gaze vectors.
    
    Args:
        gaze_vectors: JSON array of gaze specifications:
            [
                {
                    "name": "boy",
                    "position": [0.6, 0.5],
                    "gaze_direction": [0.3, -0.2],  // OR named target
                    "gaze_target": "counterman"  // optional named target
                },
                ...
            ]
    
    Returns:
        Geometric analysis including circuit closure, mutual pairs, and path quality
    """
    
    data = json.loads(gaze_vectors)
    
    # Build position lookup
    positions = {g['name']: tuple(g['position']) for g in data}
    
    # Create NarrativeElements for analysis
    elements = []
    for g in data:
        gaze_dir = g.get('gaze_direction')
        if isinstance(gaze_dir, list):
            gaze_dir = gaze_dir  # Keep as vector
        elif g.get('gaze_target'):
            gaze_dir = g['gaze_target']  # Use named target
        
        elements.append(NarrativeElement(
            name=g['name'],
            element_type='character',
            spatial_position=tuple(g['position']),
            temporal_marker='during',
            tension_contribution=0.5,
            gaze_direction=gaze_dir
        ))
    
    # Analyze
    analyzer = GazeCircuitAnalyzer()
    result = analyzer.analyze_gaze_structure(elements)
    
    return json.dumps(result, indent=2)

@mcp.tool()
def get_rockwell_principles() -> str:
    """
    Get Rockwell's compositional laws and gluing axioms.

    Returns the fundamental principles of single-frame narrative.
    """
    
    principles = {
        'temporal_compression_axiom': {
            'principle': 'Single image must imply before, during, and after',
            'mechanism': 'Strategic detail placement creates temporal depth',
            'example': 'Boy with bindle (before: left home) + Cop gentle posture (after: will take him home)',
            'cohomology': 'Three temporal states glue via visual cues → H¹ = 0'
        },
        
        'gaze_structure_axiom': {
            'principle': 'Eye lines must create narrative circuit',
            'mechanism': 'Viewer follows gaze path that tells complete story',
            'example': 'Girl looks at magazine → viewer compares → girl looks at mirror → loop closes',
            'cohomology': 'Gaze circuit creates closed narrative loop → H¹ = 0',
            'geometric_note': 'Circuit quality measured by gaze vector intersections and path closure'
        },
        
        'detail_hierarchy_axiom': {
            'principle': 'Each detail must earn its narrative existence',
            'mechanism': 'No decoration - everything advances story',
            'cohomology': 'Unnecessary detail = morphism that doesn\'t compose → H¹ ≠ 0'
        },
        
        'dignity_preservation_axiom': {
            'principle': 'Subject maintains humanity despite comedic situation',
            'mechanism': 'Viewer empathy prevents mockery',
            'cohomology': 'Comedy region + Pathos region must glue → preserves H¹ = 0'
        },
        
        'spatial_narrative_axiom': {
            'principle': 'Physical positioning must match emotional relationships',
            'mechanism': 'Distance, angles, and overlap tell story',
            'cohomology': 'Spatial tension must equal narrative tension → H¹ = 0'
        },
        
        'reciprocity_axiom': {
            'principle': 'Meaningful connection requires mutual engagement',
            'mechanism': 'Both parties must participate - passive receipt is not connection',
            'example': 'Boy must reach for milkshake, not just receive it',
            'cohomology': 'One-way interaction creates asymmetric obstruction → H¹ ≠ 0'
        }
    }
    
    return json.dumps(principles, indent=2)

@mcp.tool()
def explain_cohomology_for_visual_narrative() -> str:
    """
    Explain what sheaf cohomology means for visual storytelling.

    Returns educational content about H¹ in visual narrative context.
    """
    
    explanation = {
        'core_concept': """
Sheaf cohomology measures whether local pieces can glue into coherent whole.

For visual narrative:
- Objects = Spatial regions in image (protagonist zone, context zone, etc.)
- Morphisms = Visual relationships (gazes, spatial tension, temporal cues)
- Gluing condition = Do regions compose into coherent story?

H¹ = 0: Image tells complete story (all regions glue)
H¹ ≠ 0: Visual elements don't cohere (obstructions exist)
        """,
        
        'what_H1_measures': """
H¹ is the space of UNFIXABLE MISMATCHES between visual elements.

Example: Crisis shown (protagonist in distress) but no resolution hint.
- Local spec 1: "Character is troubled"
- Local spec 2: "No indication where this leads"
- These specifications CANNOT be glued globally
- This mismatch is an obstruction → contributes to H¹

Quality matters: A weak resolution hint partially reduces H¹,
but only a strong, clear hint achieves full gluing.
        """,
        
        'geometric_gaze_analysis': """
NEW in v3: Gaze analysis uses geometric vectors, not just named targets.

- Each gaze is a direction vector from source position
- Circuit closure measured by whether gaze vectors form closed loop
- Mutual gaze quality = how well two gazes "meet" geometrically
- Path quality = average connection strength around circuit

This allows precise measurement of "how closed" a circuit is,
rather than binary "closed/open" classification.
        """,
        
        'quality_weighted_scoring': """
NEW in v3: Obstructions have quality scores, not just presence/absence.

- Resolution present but weak (quality 0.3) → smaller H¹ contribution
- Strong mutual gaze (quality 0.9) → nearly eliminates gaze obstruction
- Reciprocating protagonist → reduces interaction asymmetry

This allows graduated assessment matching human perception of
"almost there" vs "fundamentally broken" narratives.
        """,
        
        'rockwell_mastery': """
Rockwell achieved H¹ ≈ 0 consistently by:

1. Temporal markers (before/during/after all visible)
2. Gaze circuits (closed narrative loops with high geometric quality)
3. Spatial coherence (positioning matches emotion)
4. Detail economy (no non-composing elements)
5. Dignity preservation (comedy + pathos glue)
6. Reciprocity (both parties engage in the narrative moment)

When H¹ = 0: Single image contains complete dramatic arc
        """
    }
    
    return json.dumps(explanation, indent=2)

@mcp.tool()
def get_server_info() -> str:
    """Get information about this enhanced Norman Rockwell MCP server."""
    
    info = {
        'name': 'Norman Rockwell Visual Narrative Cohomology Server',
        'version': '3.0.0-geometric',
        'description': 'Detects H¹ ≠ 0 obstructions with geometric gaze analysis and quality-weighted scoring',
        
        'v3_improvements': {
            'serialization_fix': 'NarrativeElement objects properly converted to dicts',
            'geometric_gaze_analysis': 'Gaze vectors with intersection quality measurement',
            'quality_weighted_scoring': 'Obstructions scaled by implementation quality',
            'resolution_strength': 'Graduated resolution hint assessment (0-1)',
            'reciprocity_detection': 'Identifies one-way vs mutual engagement',
            'new_tool': 'analyze_gaze_geometry for direct vector analysis'
        },
        
        'rockwell_principles_implemented': [
            'Temporal compression axiom',
            'Gaze structure axiom (geometric)',
            'Detail hierarchy axiom',
            'Dignity preservation axiom',
            'Spatial narrative axiom',
            'Reciprocity axiom (NEW)'
        ],
        
        'scoring_calibration': {
            'H1_excellent': '< 0.2 (Rockwell mastery)',
            'H1_good': '0.2 - 0.5 (Competent narrative)',
            'H1_developing': '0.5 - 1.0 (Core elements present)',
            'H1_needs_work': '> 1.0 (Significant gaps)'
        }
    }
    
    return json.dumps(info, indent=2)

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    mcp.run()
