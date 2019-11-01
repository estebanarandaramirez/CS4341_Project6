# CS4341_Project6: Bayesian Networks
## Esteban Aranda
## 10/06/19

### RUNNING:
To run this project open the terminal and cd into the appropriate directory. Once there, run 
"python Project6.py [network_file] [query_file] [num_samples]"
where [network_file] can be either 'network_option_a.txt' or 'network_option_b.txt',
[query_file] can be either 'query1.txt' or 'query2.txt', and [num_samples] specifies the 
number of samples to be used.

The program will output on the command line the calculated probabilities based on
the given network and query for both rejection sampling and weighted likelihood sampling. 

The project is built using Python 3.7. The external libraries utilized 
which are required to run the project are 'networkx' and 'numpy'.

### IMPLEMENTATION:
I implemented both options A and B for part 2 of the project. Therefore, the two sampling 
methods can be run with all combinations of the provided network-query files. 

My implementation of the data structure representing the Bayesian Network consists of 
an int representing the current node, the number of parents that node has, followed by the list 
of the parents, the conditional probability table in the form of a two-dimensional array,
a type representing if the node is an evidence variable, query variable, or unknown, and 
a value representing if the node is true, false, or unknown. 

The information for each node was then embedded inside each node of a Directed Graph 
built using 'networkx', which contains all the edges between the nodes. 