# Simulated annealing on CLIP
This project uses simulated annealing to optimize an image's CLIP similarity to a text prompt.

Images are produced by a shape layer (composed of blobs of various position and size) and a colour layer (composed of colours of various position and size).

Each mutation can alter these variables as well as adding or removing a colour or blob.

<table>
    <tr>
        <th>Prompt</th>
        <th>GIF</th>
        <th>Best</th>
    </tr>
    <tr>
        <td>Turtle</td>
        <td> <img src="https://github.com/SamuelmdLow/simulated-annealing-with-embeddings/blob/main/examples/turtle.gif?raw=true" alt="Turtle GIF" width = 360px height = 360px></td>
        <td> <img src="https://github.com/SamuelmdLow/simulated-annealing-with-embeddings/blob/main/examples/turtle.png?raw=true" alt="Turtle" width = 360px height = 360px></td>
    </tr> 
    <tr>
        <td>Frog</td> 
        <td> <img src="https://github.com/SamuelmdLow/simulated-annealing-with-embeddings/blob/main/examples/frog.gif?raw=true" alt="Frog GIF" width = 360px height = 360px></td>
        <td> <img src="https://github.com/SamuelmdLow/simulated-annealing-with-embeddings/blob/main/examples/frog.png?raw=true" alt="Frog" width = 360px height = 360px></td>
    </tr> 
    <tr>
        <td>Chicken bird</td>
        <td> <img src="https://github.com/SamuelmdLow/simulated-annealing-with-embeddings/blob/main/examples/chicken_bird.gif?raw=true" alt="Chicken bird GIF" width = 360px height = 360px></td>
        <td> <img src="https://github.com/SamuelmdLow/simulated-annealing-with-embeddings/blob/main/examples/chicken_bird.png?raw=true" alt="Chicken bird" width = 360px height = 360px></td>
    </tr> 
    <tr>
        <td>Mushroom</td>
        <td> <img src="https://github.com/SamuelmdLow/simulated-annealing-with-embeddings/blob/main/examples/mushroom.gif?raw=true" alt="Mushroom GIF" width = 360px height = 360px></td>
        <td> <img src="https://github.com/SamuelmdLow/simulated-annealing-with-embeddings/blob/main/examples/mushroom.png?raw=true" alt="Mushroom" width = 360px height = 360px></td>
    </tr> 
</table>

This can also be used to attempt to replicate an image by optimizing for the image's embedding.


<table>
    <tr>
        <th>Prompt</th>
        <th>GIF</th>
        <th>Best</th>
    </tr>
    <tr>
        <td><img src="https://github.com/SamuelmdLow/simulated-annealing-with-embeddings/blob/main/examples/edmonton.jpg?raw=true" alt="Edmonton" width = 360px height = 360px>
        <a href="https://commons.wikimedia.org/wiki/File:A_Panoramic_View_of_Downtown_Edmonton,_September_2019.jpg">Edmonton</a>
        </td>
        <td> <img src="https://github.com/SamuelmdLow/simulated-annealing-with-embeddings/blob/main/examples/edmonton-replica.gif?raw=true" alt="Edmonton GIF" width = 360px height = 360px></td>
        <td> <img src="https://github.com/SamuelmdLow/simulated-annealing-with-embeddings/blob/main/examples/edmonton-replica.png?raw=true" alt="Edmonton" width = 360px height = 360px></td>
    </tr>
    <tr>
        <td><img src="https://github.com/SamuelmdLow/simulated-annealing-with-embeddings/blob/main/examples/dandelion.jpg?raw=true" alt="Dandelion" width = 360px height = 360px>
        <a href="https://commons.wikimedia.org/wiki/File:Fleur_de_pissenlit_(Taraxacum).jpg">Dandelion</a>
        </td>
        <td> <img src="https://github.com/SamuelmdLow/simulated-annealing-with-embeddings/blob/main/examples/dandelion-replica.gif?raw=true" alt="Dandelion GIF" width = 360px height = 360px></td>
        <td> <img src="https://github.com/SamuelmdLow/simulated-annealing-with-embeddings/blob/main/examples/dandelion-replica.png?raw=true" alt="Dandelion" width = 360px height = 360px></td>
    </tr>
</table>