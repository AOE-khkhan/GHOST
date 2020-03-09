import json

def main():
    # how many numbers to gather
    target = 500
    
    n = 1   # the starting point(on consecutive scale)
    count = 2 #the number of prime in the initial list
    prime_numbers = [2, 3] #the initial prime
    prime_number_check_limit = 1 #the slice to use to validate

    # check if target is reached
    while count < target:
        # the odd number
        number = (2*n) + 1

        # the check
        if number >= prime_numbers[prime_number_check_limit]**2:
            prime_number_check_limit += 1

        # true until proven otherwise
        is_prime = True

        # prime number slice validation
        for prime_number in prime_numbers[:prime_number_check_limit]:
            if number % prime_number == 0:
                is_prime = False
                break
        
        # save if prime
        if is_prime:
            prime_numbers.append(number)
            count += 1
        
        # move to next value in the sequence
        n += 1

    # save as a json object
    with open('prime_numbers_list.json', 'w') as file:
        json.dump(prime_numbers, file)

if __name__ == "__main__":
    main()
