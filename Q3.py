import pandas as pd
import numpy as np
import numpy.random as npr
from numpy.random import Generator, PCG64


# a)
def draw_samples_a(BG, degrees):
    # Create a random number generator using the provided BitGenerator
    rng = Generator(BG)

    # Generate 200 i.i.d. draws from a Student's t-distribution with the specified degrees of freedom
    samples = rng.standard_t(df=degrees, size=200)

    return samples


# Example usage:
bg = PCG64()  # Create a BitGenerator instance
draws = draw_samples_a(
    bg, 6
)  # Generate 200 i.i.d. draws from a t-distribution with 10 degrees of freedom
print(draws)  # Print the generated samples

# b)


def draw_values_b(bg, a, r):
    # Create a random number generator using the provided BitGenerator
    rngb = Generator(bg)

    # Generate a 1-D array of values randomly drawn from a
    # If replace is True, values are drawn with replacement
    # If replace is False, values are drawn without replacement
    drawn_values_b = rngb.choice(a, size=a.shape[0], replace=r)

    return drawn_values_b


# Example usage:
bg = PCG64()
a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
drawn_values_b_t = draw_values_b(bg, a, True)
drawn_values_b_f = draw_values_b(bg, a, False)
print(drawn_values_b_f)  # Print the generated samples
print(drawn_values_b_t)  # Print the generated samples

# c)
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt


def probability_plot(s):
    # Generate a probability plot
    stats.probplot(s, dist="norm", plot=plt)

    # Set the title and labels
    plt.title("Probability  plot comparing its values to a N(0,1) distribution")
    plt.xlabel("quantiles")
    plt.ylabel("Values")

    # Show the plot
    plt.show()


# Example usage:
s = np.random.normal(
    0, 1, 2000
)  
probability_plot(s)  

# d)

from numpy.random import Generator, PCG64
import scipy.stats as stats

def bootstrap_test(bg, a, T):

    rng_d = Generator(bg)
    
    # Initialize the count of rejections
    rejections = 0
    
    # Repeat the bootstrap procedure and the Kolmogorov-Smirnov test T times
    for _ in range(T):
        # Bootstrap from a
        bootstrapped_values = rng_d.choice(a, size=a.shape[0], replace=True)
        
        # Calculate the Kolmogorov-Smirnov test statistic
        D, p = stats.kstest(bootstrapped_values, 'norm')
        
        # If the p-value is less than 0.05, reject H0
        if p < 0.05:
            rejections += 1
    
    # Calculate the fraction of the T simulations where we reject H0 at the 5% significance level
    p = rejections / T
    
    return p

# Example usage:
bg = PCG64()  
a = np.random.normal(0, 1, 2000)  
T = 1000  
p = bootstrap_test(bg, a, T)  
print(p)

# e)Explain briefly for each of the following [10 pts]:•What do your results from part (d) tell you about the size and power of the KS test?


#•Are your results from part (d) consistent with your results from part (c)?
