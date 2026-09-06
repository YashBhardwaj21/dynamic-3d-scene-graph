from scene_graph.config import SceneGraphConfig
from scene_graph.tracking.track import Track


class AdmissibilityFilter:
    """Filters subject-object pairs based on semantic roles and admissibility rules."""
    
    def __init__(self, config: SceneGraphConfig):
        self.config = config
        
        self.roles = self.config.get("roles", {
            "support_surface": ["desk", "table"],
            "container": [],
            "ordinary_object": ["monitor", "computer", "keyboard", "mouse", "telephone", 
                                "book", "cup", "pen", "paper"]
        })
        
        self.admissibility = self.config.get("admissibility", {
            "ON": {"subject": "ordinary_object", "object": "support_surface"},
            "INSIDE": {"subject": "ordinary_object", "object": "container"}
        })

    def get_role(self, class_name: str) -> str:
        """Get the semantic role of a class name."""
        for role_name, classes in self.roles.items():
            if class_name in classes:
                return role_name
        return "unknown"

    def is_admissible(self, predicate: str, subject: Track, object: Track) -> bool:
        """Check if a subject-object pair is admissible for a specific predicate."""
        if subject.object_id == object.object_id:
            return False
            
        rules = self.admissibility.get(predicate)
        if not rules:
            return True # Default to admissible if no specific rules
            
        subj_role = self.get_role(subject.class_name)
        obj_role = self.get_role(object.class_name)
        
        return subj_role == rules.get("subject") and obj_role == rules.get("object")
