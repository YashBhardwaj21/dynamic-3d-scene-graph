from scene_graph.config import SceneGraphConfig
from scene_graph.tracking.track import Track


from scene_graph.relations.inverse_algebra import INVERSE, SYMMETRIC

ALLOWED_PREDICATES = frozenset(INVERSE.keys()) | frozenset(SYMMETRIC)

class AdmissibilityFilter:
    """Filters subject-object pairs based on semantic roles and admissibility rules."""
    
    def __init__(self, config: SceneGraphConfig):
        self.config = config
        
        self.roles = {
            "support_surface": self.config.roles.support_surface,
            "container": self.config.roles.container,
            "ordinary_object": self.config.roles.ordinary_object
        }
        
        # In a real system, admissibility would also be configurable,
        # but the contract requires a strict set of 14 predicates.
        # We enforce ALLOWED_PREDICATES strictly.
        self.admissibility = {
            "ON": {"subject": "ordinary_object", "object": "support_surface"},
            "INSIDE": {"subject": "ordinary_object", "object": "container"}
        }

    def get_role(self, class_name: str) -> str:
        """Get the semantic role of a class name."""
        for role_name, classes in self.roles.items():
            if class_name in classes:
                return role_name
        return "unknown"

    def is_admissible(self, predicate: str, subject: Track, object: Track) -> bool:
        """Check if a subject-object pair is admissible for a specific predicate."""
        if predicate not in ALLOWED_PREDICATES:
            raise ValueError(f"Predicate '{predicate}' is unknown. Only the 14-predicate contract is supported.")
            
        if subject.object_id == object.object_id:
            return False
            
        rules = self.admissibility.get(predicate)
        if not rules:
            # If a predicate is in the 14 allowed but has no specific rules, it's admissible
            return True 
            
        subj_role = self.get_role(subject.class_name)
        obj_role = self.get_role(object.class_name)
        
        return subj_role == rules.get("subject") and obj_role == rules.get("object")
