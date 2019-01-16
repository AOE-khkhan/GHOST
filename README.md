# GHOST
GHOST, Generic Human Operating System Terminal. This is a general purpose artificial intelligence system able to understand the human communication and execute commands when asked to.

## Approach
  One major fact you should know is that the sytem is concept based. In Deep learning, the Neural Network tries to extract concepts and use the concepts to form and find higher level concepts in a deep and hierarchical structure, this is something like that but without the Neural networks. For now there are two major approach to this problem. Note that we are building the system one goal at a time:
  - ### Probabilistic
     This involves creating a probability distribution that maps concept to states.
     
  - ### Transformation
    This involves mapping concepts to one another as the connections develop to learn more indepth and indirect connections between concepts.
    
## Structure
  The fundamental element of the system information structure is called a *state*. *Concepts* are combination of states which represent patterns. It also contains a *context* this holds previous information and the size is adjustable. The concept size is also adjustable, the higher the concept size the more patterns it can discern.
  
## Test
  The current task for the program is to count numbers from 1 to 100. Its seems quite simple but believe me, it isn't. It first learns to count a from 1 to a set of numbers like 25, 25, 55 and 75 until it is taksed to count to 100 all on it's own. The results are quite impressive.
  
## Why Not Neural Networks?
  You might be wondering why neural networks were not used. Well, neural networks are very impressive but I personally don't see them as the foundation of building an AGI. I believe it will be used eventually maybe as a guiding system (like in a GAN) or something else but for now neural networks are not included in this model. 
