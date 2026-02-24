brojevi = []

while True:
    unos = input("Unesite broj (ili 'Done' za kraj): ")

    if unos == "Done":
        break

    try:
        broj = float(unos)
        brojevi.append(broj)
    except ValueError:
        print("Pogrešan unos! Molimo unesite broj ili 'Done'.")

if len(brojevi) == 0:
    print("Niste unijeli nijedan broj.")
else:
    print("\nNesortirana lista:", brojevi)

    print("Broj unesenih brojeva:", len(brojevi))
    print("Srednja vrijednost:", sum(brojevi) / len(brojevi))
    print("Minimalna vrijednost:", min(brojevi))
    print("Maksimalna vrijednost:", max(brojevi))

    brojevi.sort()
    print("Sortirana lista:", brojevi)