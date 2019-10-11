import random

print("Welcome to Mastermind!"); print()

while True:
    
    difficulty = input("Select difficulty (e/m/h): "); print()
    if difficulty == "exit":
        break
    elif difficulty == "h":
        print("Hard difficulty selected - 6 random numbers chosen from 0 to 9."); print()
        numbers = 6; num_diff = 9
    elif difficulty == "m":
        print("Medium difficulty selected - 5 random numbers chosen from 0 to 7."); print()
        numbers = 5; num_diff = 7
    else:
        print("Easy difficulty selected - 4 random numbers chosen from 0 to 5."); print()
        numbers = 4; num_diff = 5    
    
    comp = []
    for i in range(numbers):
        comp.append(random.randint(0, num_diff))
    answer = ''.join(str(i) for i in comp)
    
    num_guesses = []
    while True:
        guess = input("Your guess: ")
        if guess == "exit":
            break
        elif len(guess) == numbers:
            num_guesses.append("")
            correct = []
            for i in range(numbers):
                if guess[i] == answer[i]:
                    correct.append("")
            if len(correct) == numbers:
                if len(num_guesses) == 1:
                    grammar_g = "guess,"
                else:
                    grammar_g = "guesses,"
                print("You got it right in", len(num_guesses), grammar_g, "well done!"); print()
                break
            if len(correct) == 1:
                grammar_n = "number."
            else:
                grammar_n = "numbers."
            print("That guess contained", len(correct), "correct", grammar_n); print()
        else:
            print("A valid guess contains", numbers, "numbers, try again!"); print()
            continue

    if guess == "exit":
        break
    rematch = input("Would you like to play again? y/n: "); print()
    if rematch == "y":
        continue
    else:
        break
