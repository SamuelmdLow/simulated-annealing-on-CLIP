# https://www.sbert.net/examples/sentence_transformer/applications/image-search/README.html
import sys
import random
import os
import copy
import statistics
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from PIL import Image

from mutationStrategies import RandomPixelFlipStrategy, MoveBlobsStrategy, ColourStripesStrategy, ColourBlobs, ColourShapeSimultaneous, ColourInsideMask
from scoringSystems import EmbeddingsScoring
from localSearch import SimulatedAnnealing

# Colours
white = (255, 255, 255)
black = (0, 0, 0)

def generate_mutation(baseMutationStrategy, temp):
    ms = copy.deepcopy(baseMutationStrategy)
    ms.mutate_image(temp)
    image = ms.render_image() 
    return image

def generate_sample(mutationStrategy, sample_size, temp):
    start = time.time()
    baseMutationStrategy = copy.deepcopy(mutationStrategy)
    baseMutationStrategy.mutate_image(1)
    baseImage = baseMutationStrategy.render_image()
    
    
    images = [generate_mutation(baseMutationStrategy, temp) for s in range(sample_size)]

    return [baseImage, images]

def sample_distances(mutationStrategy, scoreSystem, samples=1000, sample_size = 100, temp=0.01):
    start = time.time()

    samples = [generate_sample(mutationStrategy, sample_size, temp) for sample in range(samples)]
    sampleTime = time.time()

    means = scoreSystem.compute_sample_sim_means(samples, sample_size)
    distTime = time.time()

    print(f"sample: {sampleTime-start}s, dis: {distTime-sampleTime}s")

    return means


def alternate_blobs_pixels(scoreSystem, imageDir, iterations):
    blobsStrategy = MoveBlobsStrategy(128, 128, 3, white)
    pixelsStrategy = RandomPixelFlipStrategy(128, 128)
    
    localSearch = SimulatedAnnealing(copy.deepcopy(pixelsStrategy), scoreSystem)

    for i in range(iterations):
        localSearch.search(f"images/{imageDir}/{i}/pixels", initial_temp=1/(i+1))

        blobsStrategy.colour = [white,black][i%2]
        blobsStrategy.baseImage = localSearch.best_history[-1]
        localSearch.mutationStrategy = copy.deepcopy(blobsStrategy)

        localSearch.search(f"images/{imageDir}/{i}/influence")

        pixelsStrategy.representation = localSearch.best_history[-1]
        localSearch.mutationStrategy = copy.deepcopy(pixelsStrategy)

        localSearch.save_history("alt-blobs-pixels", f"{imageDir}/")

def random_nouns(size):
    noun_file = open("data/english-nouns.txt", "r")
    nouns = noun_file.read().split("\n")
    chosen_nouns = [nouns[int(len(nouns) * random.random())] for i in range(size)]
    print(chosen_nouns)
    return chosen_nouns

def mds(name, strategies, strategyNames, sample_size=32):
    # https://www.geeksforgeeks.org/machine-learning/sklearn-multi-dimensional-scaling-mds-python-implementation-from-scratch/

    scoreSystem = EmbeddingsScoring("clip-ViT-B-32")

    # texts
    chosen_nouns = random_nouns(sample_size)

    images = []
    for strategy in strategies:
        for i in range(sample_size):
            strategy.mutate_image(1)
            images.append(strategy.render_image())

    legend = []
    for i in range(1+ len(strategies)):
        legend += [i] * sample_size
    print(legend)
    distanceMatrix = scoreSystem.distanceMatrix(chosen_nouns+images)

    mds = MDS(n_components=2)
    X_reduced = mds.fit_transform(distanceMatrix)

    # Visualize the reduced data
    plt.figure(figsize=(8, 6))

    plt.scatter(X_reduced[0:sample_size, 0], X_reduced[0:sample_size, 1], label="Text")

    for (i, strategyName) in enumerate(strategyNames):
        plt.scatter(X_reduced[(i+1) * sample_size:(i+2) * sample_size, 0], X_reduced[(i+1) * sample_size:(i+2) * sample_size, 1], label=strategyName)
    
    plt.legend(loc='best')
    plt.title("MDS Visualization of Random Images from Each Representations")
    plt.xlabel("MDS Dimension 1")
    plt.ylabel("MDS Dimension 2")

    plt.savefig(f"plots/mds--{name}.png")
    plt.show()

def plot_avg_similarity_over_parameter(name, mutationStrategiesMap, parameters, samples=256):
    scoreSystem = EmbeddingsScoring("clip-ViT-B-32")
    
    means = []

    for p in parameters:
        mutationStrategy = mutationStrategiesMap(p)
        sample = sample_distances(mutationStrategy, scoreSystem, samples=samples, sample_size=1, temp=1)
        means.append(statistics.mean(sample))
        print(f"{p}) mean: {statistics.mean(sample)}, stdev: {statistics.stdev(sample)}")

    plt.plot(parameters, means)
    plt.title(f"Average similarity over {name}")
    plt.xlabel(f"{name}")
    plt.ylabel("CLIP similarity")
    plt.grid(axis = 'x')

    plt.savefig(f"plots/{name.replace(' ', '_')}--avg_sim.png")
    plt.show()

    print(means)

def plot_annealing_over_parameter(name, mutationStrategiesMap, parameters, iterations=500):

    scoreSystem = EmbeddingsScoring("clip-ViT-B-32")

    nouns = random_nouns(4)

    avg_scores = []
    for p in parameters:
        scores = []
        for noun in nouns:
            scoreSystem.set_goal_text(noun)
            mutationStrategy = mutationStrategiesMap(p)
            localSearch = SimulatedAnnealing(mutationStrategy, scoreSystem)
            localSearch.search(alpha=0.99, max_iterations=iterations)
            scores.append(localSearch.best_score)
        avg_scores.append(np.mean(scores))

    plt.plot(parameters, avg_scores)
    plt.title(f"Average similarity after {iterations} iterations over {name}")
    plt.xlabel(f"{name}")
    plt.ylabel("CLIP similarity")
    plt.grid(axis = 'x')

    plt.savefig(f"plots/{name.replace(' ', '_')}--annealing.png")

def race(name, strategies, strategyNames, prompt=None, alpha=0.99, iterations=500):

    scoreSystem = EmbeddingsScoring("clip-ViT-B-32")

    if prompt == None:
        prompt = random_nouns(1)[0]

    for i, strategy in enumerate(strategies):
        scores = []

        scoreSystem.set_goal_text(prompt)
        localSearch = SimulatedAnnealing(strategy, scoreSystem)
        localSearch.search(alpha=alpha, max_iterations=iterations)
        localSearch.save_history(strategyNames[i], f"race/{name}-{prompt}")

        plt.plot(localSearch.best_points_iteration, localSearch.best_points_score, label=strategyNames[i])
        plt.scatter(localSearch.best_points_iteration, localSearch.best_points_score)
    
    plt.title(f"Comparison of annealing for word '{prompt}' after {iterations} iterations")
    plt.xlabel(f"Iteration")
    plt.ylabel("CLIP similarity")
    plt.grid(axis = 'x')
    plt.legend(loc='best')

    plt.savefig(f"plots/annealing_race--{name}-{prompt}.png")

def anneal_boxplot(name, strategies, strategyNames, iterations=500, sample_size=16):
    scoreSystem = EmbeddingsScoring("clip-ViT-B-32")

    nouns = random_nouns(sample_size)

    strategyScores = []
    for i, strategy in enumerate(strategies):
        scores = []
        for noun in nouns:
            scoreSystem.set_goal_text(noun)
            localSearch = SimulatedAnnealing(strategy, scoreSystem)
            localSearch.search(alpha=0.99, max_iterations=iterations)
            scores.append(localSearch.best_score)
        strategyScores.append(scores)
    
    plt.boxplot(strategyScores, labels=strategyNames)

    plt.title(f"Similarity after {iterations} iterations")
    plt.xlabel(f"{name}")
    plt.ylabel("CLIP similarity")
    plt.grid(axis = 'x')

    plt.savefig(f"plots/{name.replace(' ', '_')}--boxplot.png")

def simmilarity_boxplot(name, strategies, strategyNames, sample_size=128):
    scoreSystem = EmbeddingsScoring("clip-ViT-B-32")

    strategyScores = []
    for strategy in strategies:
        sample = sample_distances(strategy, scoreSystem, samples=sample_size, sample_size=1, temp=1)
        strategyScores.append(sample)
    
    plt.boxplot(strategyScores, labels=strategyNames)

    plt.title(f"Comparison of similarity distributions within each representation")
    plt.xlabel(f"{name}")
    plt.ylabel("CLIP similarity")
    plt.grid(axis = 'x')

    plt.savefig(f"plots/{name.replace(' ', '_')}--similarity-boxplot.png")


if __name__ == '__main__':
    command = sys.argv[1]

    if command == "anneal":
        mutationOption = sys.argv[2]
        prompt = sys.argv[3]
        imageDir = prompt.replace(".", "_").replace("/", "-")

        scoreSystem = EmbeddingsScoring("clip-ViT-B-32")

        if ".jpg" in prompt or ".png" in prompt:
            scoreSystem.set_goal_image(Image.open(prompt))
        else:
            scoreSystem.set_goal_text(prompt.replace("_", " "))

        path = os.path.join("images", imageDir)
        os.makedirs(path, exist_ok=True)
        
        if mutationOption == "alternate":
            # anneal alternate PROMPT
            alternate_blobs_pixels(scoreSystem, imageDir, 10)

        elif mutationOption == "increasing-blob":
            # anneal increasing-blob PROMPT ITERATIONS GROUPSIZE
            iterations = int(sys.argv[4])
            groupSize = int(sys.argv[5])
            mutationStrategy = ColourShapeSimultaneous(128, 128, colourBlobCount=groupSize, shapeBlobCount=groupSize)                
            localSearch = SimulatedAnnealing(copy.deepcopy(mutationStrategy), scoreSystem)

            for i in range(iterations):
                localSearch.search(alpha=1 - 0.1/(i+1), initial_temp=1 - (i/iterations)**2, image_path=imageDir)
                localSearch.save_history("increasing-blob", f"{imageDir}/")
                
                for blob in localSearch.mutationStrategy.shape.representation:
                    blob.tempAdjust = blob.tempAdjust * 0.75
                for blob in localSearch.mutationStrategy.colour.representation:
                    blob.tempAdjust = blob.tempAdjust * 0.75

                for g in range(groupSize):
                    localSearch.mutationStrategy.shape.add_blob()
                    localSearch.mutationStrategy.colour.add_blob()

        elif mutationOption == "coloured-blob":
            # anneal coloured-blob PROMPT BLOBCOUNT STRIPECOUNT
            blobCount = 3
            if len(sys.argv) > 4:
                blobCount = int(sys.argv[4])
            blobsStrategy = MoveBlobsStrategy(128, 128, blobCount, white, recenter=True)

            stripeCount = 5
            if len(sys.argv) > 5:
                stripeCount = int(sys.argv[5])
            stripesStrategy = ColourStripesStrategy(128, 128, stripeCount)

            localSearch = SimulatedAnnealing(blobsStrategy, scoreSystem)
            localSearch.search(alpha=0.99)
            localSearch.save_history("coloured-blob--shape", f"{imageDir}/")
            mask = blobsStrategy.as_mask(localSearch.best_representation)

            localSearch = SimulatedAnnealing(stripesStrategy, scoreSystem)
            localSearch.search(alpha=0.99)
            localSearch.save_history("coloured-blob--colour", f"{imageDir}/")
            pallet = copy.deepcopy(localSearch.best_representation)
            
            colouringStrategy = ColourInsideMask(128, 128, mask, pallet=pallet, blobPerColour=2)
            localSearch = SimulatedAnnealing(colouringStrategy, scoreSystem)
            localSearch.search(alpha=0.99)
            localSearch.save_history("coloured-blob", f"{imageDir}/")

        elif mutationOption == "fixed-colour-blob":
            # anneal fixed-colour-blob PROMPT STRIPECOUNT BLOBCOUNT
            stripeCount = 5
            if len(sys.argv) > 4:
                stripeCount = int(sys.argv[4])
            stripesStrategy = ColourStripesStrategy(128, 128, stripeCount)

            localSearch = SimulatedAnnealing(stripesStrategy, scoreSystem)
            localSearch.search(alpha=0.99)
            localSearch.save_history("coloured-blob--colour", f"{imageDir}/")
            pallet = copy.deepcopy(localSearch.best_representation)
            
            blobCount = 5
            if len(sys.argv) > 5:
                blobCount = int(sys.argv[5])
            blobsStrategy = ColourShapeSimultaneous(128, 128, shapeBlobCount=blobCount, pallet=pallet)

            localSearch = SimulatedAnnealing(blobsStrategy, scoreSystem)
            localSearch.search(alpha=0.99)
            localSearch.save_history("coloured-blob", f"{imageDir}/")

        else:
            if mutationOption == "blob":
                # anneal blob PROMPT ("free"|BLOBCOUNT)
                blobCount = 3
                freeBlobCount = False
                if len(sys.argv) > 4:
                    if sys.argv[4] == "free":
                        freeBlobCount = True
                    else:
                        blobCount = int(sys.argv[4])

                mutationStrategy = MoveBlobsStrategy(128, 128, blobCount, white, recenter=True, freeBlobCount=freeBlobCount)

            elif mutationOption == "pixel":
                # aneal pixel PROMPT
                mutationStrategy = RandomPixelFlipStrategy(128, 128)

            elif mutationOption == "stripes":
                # anneal stripes PROMPT STRIPECOUNT
                stripeCount = 3
                if len(sys.argv) > 4:
                    stripeCount = int(sys.argv[4])

                mutationStrategy = ColourStripesStrategy(128, 128, stripeCount)

            elif mutationOption == "colourShape":
                # anneal colourShape PROMPT COLOURSBLOBCOUNT SHAPEBLOBCOUNT
                coloursBlobCount = 5
                shapeBlobCount = 5
                freeBlobCount = True
                if len(sys.argv) > 4:
                    coloursBlobCount = int(sys.argv[4])
                    freeBlobCount = False               
                if len(sys.argv) > 5:
                    shapeBlobCount = int(sys.argv[5])
                
                mutationStrategy = ColourShapeSimultaneous(128, 128, colourBlobCount=coloursBlobCount, shapeBlobCount=shapeBlobCount, freeBlobCount=freeBlobCount)

            elif mutationOption == "colourBlob":
                # anneal colourBlob PROMPT COLOURSBLOBCOUNT SHAPEBLOBCOUNT
                blobCount = 5
                freeBlobCount = True
                if len(sys.argv) > 4:
                    blobCount = int(sys.argv[4])
                    freeBlobCount = False               
                
                mutationStrategy = ColourBlobs(128, 128, blobCount=blobCount, freeBlobCount=freeBlobCount)

            localSearch = SimulatedAnnealing(mutationStrategy, scoreSystem)
            localSearch.search(alpha=0.995, max_iterations=3000, image_path=imageDir)
            localSearch.save_history(mutationOption, f"{imageDir}/")


    elif command == "measure_steps":
        scoreSystem = EmbeddingsScoring("clip-ViT-B-32")
        
        mutationStrategy = RandomPixelFlipStrategy(128, 128)
        
        if sys.argv[2] == "blob":
            mutationStrategy = MoveBlobsStrategy(128, 128, 3, white)
        elif sys.argv[2] == "stripes":
            mutationStrategy = ColourStripesStrategy(128, 128, 5)

        means = []
        temp = 1

        xpoints = np.arange(21) * 0.05
        for temp in xpoints:
            
            sample = sample_distances(mutationStrategy, scoreSystem, samples=64, sample_size=1, temp=temp)
            means.append(statistics.mean(sample))
            print(f"{temp}) mean: {statistics.mean(sample)}, stdev: {statistics.stdev(sample)}")

        plt.plot(xpoints, means)
        plt.title(f"{sys.argv[2]} mutation strategy")
        plt.xlabel("Step size")
        plt.ylabel("CLIP similarity")
        
        plt.savefig(f"plots/measure_steps--{sys.argv[2]}.png")
        plt.show()

        print(means)

    elif command == "stripe_avg_over_p":
        mutationStrategiesMap = lambda p: ColourStripesStrategy(128, 128, p)
        counts = np.arange(40) + 1
        
        plot_avg_similarity_over_parameter("Stripe Count", mutationStrategiesMap, counts)

    elif command == "stripe_annealing_over_p":
        mutationStrategiesMap = lambda p: ColourStripesStrategy(128, 128, p)
        counts = [1, 2, 3, 4, 5, 6, 7, 8]

        plot_annealing_over_parameter("Stripe Count", mutationStrategiesMap, counts)

    elif command == "blob_avg_over_p":
        mutationStrategiesMap = lambda p: MoveBlobsStrategy(128, 128, p, white, recenter=True)
        counts = [1, 2, 3, 4, 5, 6, 7, 8]

        plot_avg_similarity_over_parameter("Blob Count", mutationStrategiesMap, counts)

    elif command == "blob_annealing_over_p":
        mutationStrategiesMap = lambda p: MoveBlobsStrategy(128, 128, p, white, recenter=True)
        counts = [1, 6, 11, 16, 21, 26]

        plot_annealing_over_parameter("Blob Count", mutationStrategiesMap, counts)

    elif command == "compare":
        strategyOption = sys.argv[2]
        comparisonOption = sys.argv[3]

        if strategyOption == "all":
            strategies = [
                RandomPixelFlipStrategy(128, 128),
                MoveBlobsStrategy(128, 128, 4, white, recenter=True),
                ColourStripesStrategy(128, 128, 4),
                ColourBlobs(128, 128, 4),
                ColourShapeSimultaneous(128, 128),
            ]
            names = [
                "Black-white pixels",
                "Blobs",
                "Stripes",
                "Colour boundaries",
                "Colour shape"
            ]

        elif strategyOption == "stripes":
            counts = [1, 2, 3, 4, 5, 6, 7, 8]
            strategies = [ColourStripesStrategy(128, 128, count) for count in counts]
            names = [f"{count} stripes" for count in counts]

        elif strategyOption == "blobs":
            counts = [1, 2, 3, 4, 5, 6, 7, 8]
            strategies = [MoveBlobsStrategy(128, 128, count, white, recenter=True) for count in counts]
            names = [f"{count} blobs" for count in counts]

            strategies.append(MoveBlobsStrategy(128, 128, 5, white, recenter=True, freeBlobCount=True))
            names.append("free blob count")

        if comparisonOption == "mds":
            mds(strategyOption, strategies, names)
        
        elif comparisonOption == "race":
            prompt = sys.argv[4]
            race(strategyOption, strategies, names, prompt=prompt, alpha=0.995, iterations=1000)

        elif comparisonOption == "anneal_boxplot":
            anneal_boxplot(strategyOption, strategies, names, iterations=500)

        elif comparisonOption == "similarity_boxplot":
            simmilarity_boxplot(strategyOption, strategies, names)

    else:
        print("Options:\n   * anneal {alternate/increasing-blob/blob/pixel} {prompt}\n   * measure_steps {pixel/blob}")
