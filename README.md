# JetsonAutonomousDriving

## Table of contents

1. [Presentation](#presentation)
2. [Project architecture](#project-architecture)

## Presentation

<p><img width="620" src="img.png"></p>

>The objective of this project is to explore the possibilities of NVIDIA's Jetson Nano card for real-time video processing. The idea is to develop a processing chain for the autonomous driving of a small robot-car.

Weekly report : https://fr.overleaf.com/project/61f3ec98ecde9c67e26d5388

## Project architecture

<pre><code>
JetsonAutonomousDriving/
      ├── src/                   
      |    ├── tutorial/                (Folder containing CNN/PyTorch tutorials)
      |    └── main/              
      |         ├── model_benchmark/    (Benchmarks of different CNN on Jetson card) 
      |         |    ├── models/        (Folder containing .pth files for each model tested)
      |         |    └── results.txt    (File containing some results and a link to google sheets)
      |         └── model/              (Best model for our project)
      ├── txt/                   
      |    ├── subject.pdf              (Original subject (in french))
      |    └── todo.txt                 (Project todolist)
      ├── assets/ 	                (Additional assets needed for the projects such as Jetson SDK) 
      ├── README.md		          
      └── LICENSE  
</pre></code>
