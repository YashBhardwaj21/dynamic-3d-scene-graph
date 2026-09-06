from typing import Dict

from scene_graph.config import SceneGraphConfig
from scene_graph.tracking.track import Track
from scene_graph.relations.inverse_algebra import INVERSE, SYMMETRIC


ALLOWED_PREDICATES = frozenset(INVERSE) | frozenset(SYMMETRIC)


class AdmissibilityFilter:

    def __init__(self, config: SceneGraphConfig):
        self.config = config

        self.roles: Dict[str, set[str]] = {
            "support_surface": set(config.roles.support_surface),
            "container": set(config.roles.container),
            "ordinary_object": set(config.roles.ordinary_object),
        }

        self.admissibility = {
            predicate: {
                "subject": rule.subject,
                "object": rule.object,
            }
            for predicate, rule in config.relations.admissibility.items()
        }

    def get_role(self, class_name: str) -> str:
        matches = [
            role
            for role, classes in self.roles.items()
            if class_name in classes
        ]

        if len(matches) > 1:
            raise ValueError(
                f"Class '{class_name}' belongs to multiple semantic roles: "
                f"{matches}"
            )

        return matches[0] if matches else "unknown"

    def _get_rule(self, predicate: str):
        rule = self.admissibility.get(predicate)

        if rule is not None:
            return rule

        inverse = INVERSE.get(predicate)

        if inverse is not None:
            inverse_rule = self.admissibility.get(inverse)

            if inverse_rule is not None:
                return {
                    "subject": inverse_rule["object"],
                    "object": inverse_rule["subject"],
                }

        return None

    def is_admissible(
        self,
        predicate: str,
        subject: Track,
        object: Track,
    ) -> bool:

        if predicate not in ALLOWED_PREDICATES:
            raise ValueError(
                f"Predicate '{predicate}' is not part of the "
                f"14-predicate relation contract."
            )

        if subject.object_id == object.object_id:
            return False

        rule = self._get_rule(predicate)

        if rule is None:
            return True

        subject_role = self.get_role(subject.class_name)
        object_role = self.get_role(object.class_name)

        return (
            subject_role == rule["subject"]
            and object_role == rule["object"]
        )