spam_count = 0
ham_count = 0

spam_word_total = 0
ham_word_total = 0

spam_exclamation_count = 0

try:
    with open("SMSSpamCollection.txt", "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            parts = line.split()

            if len(parts) == 0:
                continue

            label = parts[0]
            message_words = parts[1:]
            word_count = len(message_words)

            if label == "spam":
                spam_count += 1
                spam_word_total += word_count

                if line.endswith("!"):
                    spam_exclamation_count += 1

            elif label == "ham":
                ham_count += 1
                ham_word_total += word_count

    if spam_count > 0:
        avg_spam = spam_word_total / spam_count
    else:
        avg_spam = 0

    if ham_count > 0:
        avg_ham = ham_word_total / ham_count
    else:
        avg_ham = 0

    print("Prosječan broj riječi u spam porukama:", avg_spam)
    print("Prosječan broj riječi u ham porukama:", avg_ham)
    print("Broj spam poruka koje završavaju uskličnikom:", spam_exclamation_count)

except FileNotFoundError:
    print("Datoteka 'SMSSpamCollection.txt' nije pronađena.")