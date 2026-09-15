import time
import numpy as np

# Scoring systems
class ScoringSystem():
    def score_image(self, image):
        return 0

class EmbeddingsScoring():
    def __init__(self, model):
        from sentence_transformers import SentenceTransformer 
        # Load CLIP model
        self.model = SentenceTransformer(model)
        self.goal = None
        
    def set_goal_text(self, text):
        # Encode text descriptions
        self.goal = self.model.encode([text])

    def set_goal_image(self, image):
        # Encode image
        self.goal = self.model.encode(image)

    def score_image(self, image):
        # Compare image embedding to goal
        img_emb = self.model.encode(image)

        similarity_score = self.model.similarity(img_emb, self.goal).tolist()[0]

        return similarity_score[0]

    def score_images(self, images):
        # Compare multiple images to goal
        images = self.model.encode(images)
        similarity_score = self.model.similarity(images, self.goal).tolist()

        return similarity_score
    
    def compare_images_to_image(self, images, image):
        image = self.model.encode(image)
        images = self.model.encode(images)
        similarity_score = self.model.similarity(image, images).tolist()

        return similarity_score

    def compute_sample_sim_means(self, samples, sample_size):
        start = time.time()
        flat_list = []
        for sample in samples:
            flat_list.append(sample[0])
            flat_list.extend(sample[1])

        encodings = self.model.encode(flat_list, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
        encodingTime = time.time()

        means = []
        setLen = sample_size + 1
        for i in range(len(samples)):
            index = i * setLen

            baseEncoding = encodings[index]
            sampleEncoding = encodings[index + 1: index+setLen]
            means.append(np.mean(self.model.similarity(baseEncoding, sampleEncoding).tolist()[0]))
        simTime = time.time()

        print(f"encoding: {encodingTime-start}, sim: {simTime-encodingTime}")
        
        return means

    def distanceMatrix(self, content):
        encodings = self.model.encode(content)
        return self.model.similarity(encodings, encodings)
