#Aqsa Noreen
import os

def get_output_of_file(filename):
    return os.popen(f'python {filename}').read()

if __name__ == "__main__":
    # mdp = get_output_of_file("mdp.py")
    # pass all of the tests

    approach = get_output_of_file("Approach.py")
    print("The outputs for Approach.py:")
    print(approach)
# Question 1 part 1
    blackjack_output = get_output_of_file("blackjack.py")
    print("The outputs for blackjack.py:")
    print(blackjack_output)
# question 2 part 2
    play_blackjack_output = get_output_of_file("play_blackjack.py")
    print("\nThe outputs for play_blackjack.py:")
    print(play_blackjack_output)
