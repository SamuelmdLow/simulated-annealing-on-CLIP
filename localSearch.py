from PIL import Image
import random
import os
import math
import copy

# Local search methods
class LocalSearchMethod():
    def __init__(self, mutationStrategy, scoringSystem):
        self.mutationStrategy = mutationStrategy
        self.scoringSystem = scoringSystem

        self.score = 0
        self.best_score = 0

        self.history = []
        self.best_history = []
        self.best_representation = None
        
        self.best_points_iteration = []
        self.best_points_score = []

    def update_best(self, iteration, representation=None, score=None, image=None, image_path=None):
        if score == None:
            score = self.score

        if score >= self.best_score:
            if representation == None:
                representation = copy.deepcopy(self.mutationStrategy.representation)
            if image == None:
                image = self.mutationStrategy.render_image()

            self.best_representation = representation
            self.best_score = score
            self.best_history.append(image)

            self.best_points_iteration.append(iteration)
            self.best_points_score.append(score)

            if image_path:
                image.save(f"images/{image_path}/best.png","PNG")

    def save_history(self, name, path):

        os.makedirs(f"images/{path}", exist_ok=True)

        self.history[0].save(f"images/{path}/{name}_GIF.gif", 
                save_all = True, append_images = self.history[1:], 
                optimize = False, duration = 10) 
        
        self.best_history[0].save(f"images/{path}/{name}_bestGIF.gif", 
                save_all = True, append_images = self.best_history[1:], 
                optimize = False, duration = 10)

        self.best_history[-1].save(f"images/{path}/{name}_best.png", "PNG")

    def search(self, image_path=None):
        pass

class SimulatedAnnealing(LocalSearchMethod):
    def __init__(self, mutationStrategy, scoringSystem):
        super().__init__(mutationStrategy, scoringSystem)

    def search(self, image_path=None, alpha=0.95, initial_temp=1, min_temp=0.0000001, max_iterations=None, satisfying_score=1, harden_trigger=None):
        MIN_DELTA = 0.005
        MAX_NO_CHANGE = 15
        RESTART_TRIGGER = 0.05

        if image_path:
            os.makedirs(f"images/{image_path}", exist_ok=True)        

        temp = initial_temp

        mutationStrategy = self.mutationStrategy
        image = mutationStrategy.render_image()
        self.score = self.scoringSystem.score_image(image)
        self.update_best(0, image=image, image_path=image_path)

        initial_score = self.best_score

        no_change_count = 0
        i = 0

        harden_i = 0

        while temp > min_temp and self.best_score < satisfying_score:         
            previousRepresentation = copy.deepcopy(mutationStrategy.representation)

            mutationStrategy.mutate_image(temp)

            score = self.scoringSystem.score_image(mutationStrategy.render_image())

            score_dif = 100 * (score - self.score)

            if score_dif < -MIN_DELTA and temp > 0:
                # accept worse random step
                random_prob = math.exp(score_dif/temp)
            else:
                # optimum step
                random_prob = 1
            
            at_max = MAX_NO_CHANGE and no_change_count > MAX_NO_CHANGE

            if random.random() < random_prob or (at_max and RESTART_TRIGGER > self.best_score - score):
                # Take step
                self.score = score
                no_change_count = 0
            elif at_max:
                self.score = self.best_score
                mutationStrategy.representation = copy.deepcopy(self.best_representation)
                no_change_count = 0
            else:
                # Revert step
                mutationStrategy.representation = previousRepresentation
                no_change_count = no_change_count + 1

            temp = temp * alpha

            i = i + 1

            image = mutationStrategy.render_image()
            self.history.append(image)
            
            if self.score > self.best_score:
                self.update_best(i, image=image, image_path=image_path)
                print(f"{i} {self.best_score}")
            
            if max_iterations and i >= max_iterations:
                break

            if harden_trigger and i-harden_i >= harden_trigger:
                print(f"Harden {self.best_score} / {i-harden_i}")
                harden_i = i
                mutationStrategy.baseImage = self.best_history[-1]
                #mutationStrategy.reset_representation()
                #self.best_representation = copy.deepcopy(mutationStrategy.representation)
                temp = initial_temp

        mutationStrategy.representation = copy.deepcopy(self.best_representation)
        
        print(f'{round(((self.best_score-initial_score)/initial_score) * 100, 4)}% improvement\n  - {initial_score}\n  - {self.best_score}')


def final_refinement(image_array, pallet, scoringSystem, path):   
    width = image_array.shape[0]
    height = image_array.shape[1]
    score = scoringSystem.score_image(Image.fromarray(image_array))

    history = []
    while True:
        for y in range(height):
            for x in range(width):

                adjacent = [image_array[y][x]]

                notArrayInArray = lambda element, array: not True in [np.array_equal(element, e) for e in array]

                if x > 0:
                    if notArrayInArray(image_array[y][x-1], adjacent):
                        adjacent.append(image_array[y][x-1])

                if x < width-1:
                    if notArrayInArray(image_array[y][x+1], adjacent):
                        adjacent.append(image_array[y][x+1])

                if y > 0:
                    if notArrayInArray(image_array[y-1][x], adjacent):
                        adjacent.append(image_array[y-1][x])

                if y < height-1:
                    if notArrayInArray(image_array[y+1][x], adjacent):
                        adjacent.append(image_array[y+1][x])

                if len(adjacent) > 1:
                    attempts = []
                    for colour in adjacent:
                        newImage = copy.deepcopy(image_array)
                        newImage[y][x] = colour
                        attempts.append(newImage)

                    renderedAttempts = [Image.fromarray(img) for img in attempts]
                    similarities = scoringSystem.score_images(renderedAttempts)
                    
                    bestIndex = np.argmax(similarities)
                    image_array = attempts[bestIndex]
                    renderedAttempts[bestIndex].save(f"images/{path}/best.png", "PNG")
                    history.append(renderedAttempts[bestIndex])

                    currentScore = similarities[bestIndex][0]
                    print(f"{x},{y}: {currentScore} {pallet[bestIndex]}")
        
        print(f"{((currentScore-score)/score)*100}%,  {score} to {currentScore}")

        history[0].save(f"images/{path}/refinementGIF.gif", 
        save_all = True, append_images = history[1:], 
        optimize = False, duration = 10)

        if currentScore == score:
            break

        score = currentScore 