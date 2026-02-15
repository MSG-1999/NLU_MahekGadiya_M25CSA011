#Name : Mahek Gadiya
#Roll : M25CSA011
#Subject : NLU
#Problem No : 1

# -----------------------------------------------------

# 1) Imports & Log Setup

import re
from datetime import date

# Open a log file to save all conversations &
#'a' mode is used so previous runs are not deleted
log_file = open("M25CSA011.log", "a")

# -----------------------------------------------------

# 2) Helper Functions for Logging

def log_print(text):
    """
    This function prints chatbot output on screen
    and also writes the same output into the log file.
    """
    print(text)
    log_file.write(text + "\n")


def log_input(prompt):
    """
    This function takes input from the user
    and saves the user input into the log file.
    """
    user_text = input(prompt)
    log_file.write("You: " + user_text + "\n")
    return user_text.strip()

# -----------------------------------------------------

# 3) Age Calculation Function

def calculate_age(year, month, day):
    """
    This function calculates the user's age
    using their DOB and today's date.
    """
    today = date.today()
    age = today.year - year

    # If birthday has not happened yet this year,
    # subtract 1 from age
    if (today.month, today.day) < (month, day):
        age -= 1

    return age

# -----------------------------------------------------

# 4) Birthday Parsing Function

def parse_birthday(text):
    """
    This function tries to understand the birthday entered by the user.
    It supports multiple formats and also checks for invalid dates.
    """

    # Match numeric date formats like 12-08-2002 or 02/18/03
    numeric = re.search(r'(\d{1,2})[-/](\d{1,2})[-/](\d{2,4})', text)

    if numeric:
        a, b, y = numeric.groups()
        a, b, y = int(a), int(b), int(y)

        # Handle 2-digit year (example: 03 -> 2003)
        if y < 100:
            current_year = date.today().year % 100
            if y <= current_year:
                y += 2000
            else:
                y += 1900

        # If both values are <= 12, date is ambiguous
        if a <= 12 and b <= 12:
            return ("AMBIGUOUS", a, b, y)

        # Decide which is day and which is month
        if a > 12:
            day, month = a, b
        else:
            day, month = b, a

        # Simple validation for day and month
        if month < 1 or month > 12 or day < 1 or day > 31:
            return None

        return day, month, y

    # Match dates with month names like "15 Mar 1999"
    text_month = re.search(
        r'(\d{1,2})\s*(jan|january|feb|february|mar|march|apr|april|may|'
        r'jun|june|jul|july|aug|august|sep|september|oct|october|'
        r'nov|november|dec|december)\s*(\d{4})',
        text.lower()
    )

    if text_month:
        d, m, y = text_month.groups()

        # Convert month name to number
        month_map = {
            'jan': 1, 'january': 1,
            'feb': 2, 'february': 2,
            'mar': 3, 'march': 3,
            'apr': 4, 'april': 4,
            'may': 5,
            'jun': 6, 'june': 6,
            'jul': 7, 'july': 7,
            'aug': 8, 'august': 8,
            'sep': 9, 'september': 9,
            'oct': 10, 'october': 10,
            'nov': 11, 'november': 11,
            'dec': 12, 'december': 12
        }

        day = int(d)
        month = month_map[m]
        year = int(y)

        # Simple validation
        if day < 1 or day > 31:
            return None

        return day, month, year

    # If no format matches
    return None

# -----------------------------------------------------

# 5) Mood Detection Function ( Positive, negative, mixed )

def detect_mood(text):
    text = text.lower()

    has_positive = re.search(r'(happ|good|fine|ok|well)', text)
    has_negative = re.search(r'(sad|tir|angr|stress|bad)', text)

    # If both positive and negative words are present
    if has_positive and has_negative:
        return "mixed"

    if has_positive:
        return "positive"

    if has_negative:
        return "negative"

    return "unknown"

# -----------------------------------------------------

#6) Main Chatbot Conversation

def run_chatbot():
    """
    This function will run for one complete conversation
    with the chatbot.
    """

    log_print("\nReggy++: Hello. My name is Reggy++.")
    log_print("Reggy++: Please enter your full name.")

    # Ask name until user enters something
    # Ask name until user enters a valid alphabetic name
    while True:
         name_input = log_input("You: ")

         if not name_input:
           log_print("Reggy++: Name cannot be empty. Please try again.")
           continue

         # Extract alphabetic words only
         name_parts = re.findall(r'[A-Za-z]+', name_input)

         # If no alphabets found (e.g. 1234, @@)
         if not name_parts:
          log_print("Reggy++: Name should contain alphabets only. Please enter again.")
          continue

         # Valid name found → exit loop
         break


    # First word is first name, last word is surname
    name_parts = re.findall(r'[A-Za-z]+', name_input)
    first_name = name_parts[0]
    surname = name_parts[-1]

    log_print(f"Reggy++: Thanks, {first_name}. Your surname appears to be {surname}.")
    log_print("Reggy++: Enter your date of birth.")

    # Ask for birthday
    while True:
        dob_input = log_input("You: ")
        result = parse_birthday(dob_input)

        if result is None:
            log_print("Reggy++: Invalid date format,please try aain")
            log_print("Reggy++: Allowed formats: dd-mm-yyyy, mm-dd-yy, dd/mm/yy, dd Mon yyyy")
            log_print("Reggy++: Day Range [01-31], Month Range:[1-12].\n")
            log_print("Regg++: Please enter a valid date.")
            continue

        # Handle ambiguous dates
        if result[0] == "AMBIGUOUS":
            _, a, b, y = result
            log_print("Reggy++: Your date is ambiguous.")
            log_print("Reggy++: Enter 'dd' for day-month-year or 'mm' for month-day-year.")

            while True:
                choice = log_input("You (dd/mm): ").lower()
                if choice == "dd":
                    day, month = a, b
                    break
                elif choice == "mm":
                    day, month = b, a
                    break
                else:
                    log_print("Reggy++: Please type only 'dd' or 'mm'.")

            age = calculate_age(y, month, day)
            log_print(f"Reggy++: You are {age} years old.")
            break

        # Normal date case
        d, m, y = result
        age = calculate_age(y, m, d)
        log_print(f"Reggy++: You are {age} years old.")
        break

    # Ask about mood Postive_Negative_mixed
    log_print("Reggy++: How are you feeling today?")
    while True:
        mood_input = log_input("You: ")
        mood = detect_mood(mood_input)

        if mood == "positive":
            log_print("Reggy++: Glad to hear that.")
            break
        elif mood == "negative":
            log_print("Reggy++: Sorry to hear that.")
            break
        elif mood =="mixed":
            log_print("Reggy++: its Ambiguous.")
            log_print("Reggy++: You seem to have mixed feelings.")  
            break  
        else:
            log_print("Reggy++: I could not understand that. Please enter your mood like")
            log_print("Reggy++: happy, good, fine, ok, sad, tired, angry, stress")

    log_print("Reggy++: Conversation ended.")
    log_print("------------------------")
    
# -----------------------------------------------------

# 7) Re-Run Chatbot (for New input)

# Counter to track how many times chatbot runs
run_count = 0

# This loop allows the chatbot to run multiple times
while True:
    run_count += 1   # increase run count

    log_print(f"\n========== RUN {run_count} ==========")
    run_chatbot()

    log_print("Reggy++: Do you want to talk again? (yes/no)")
    choice = log_input("You: ").lower()

    if not re.search(r'^(yes|y)$', choice):
        log_print("Reggy++: Goodbye!")
        break

# Close the log file at the end
log_file.close()