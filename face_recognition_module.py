
import numpy as np
from insightface.app import FaceAnalysis


class FaceRecognizer:
    def __init__(self):
        self.app = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"]
        )
        self.app.prepare(
            ctx_id=-1,
            det_size=(640, 640)
        )

    def get_embedding(self, face_image):
        """
        Generate a numerical embedding for a cropped face image.
        Returns None if no face is detected.
        """
        if face_image is None or face_image.size == 0:
            return None

        faces = self.app.get(face_image)

        if not faces:
            return None

        # Use the embedding of the largest detected face.
        face = max(
            faces,
            key=lambda item: (
                item.bbox[2] - item.bbox[0]
            ) * (
                item.bbox[3] - item.bbox[1]
            )
        )

        embedding = face.normed_embedding
        return np.asarray(embedding, dtype=np.float32)
