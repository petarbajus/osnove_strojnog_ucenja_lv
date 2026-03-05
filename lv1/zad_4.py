rijeci = {}

try:
    with open("song.txt", "r", encoding="utf-8") as file:
        for red in file:
            red = red.lower().strip()
            lista_rijeci = red.split()

            for rijec in lista_rijeci:
                rijec = rijec.strip(".,!?;:\"()")

                if rijec in rijeci:
                    rijeci[rijec] += 1
                else:
                    rijeci[rijec] = 1

    jednom = [kljuc for kljuc, vrijednost in rijeci.items() if vrijednost == 1]

    print("Ukupan broj različitih riječi:", len(rijeci))
    print("Broj riječi koje se pojavljuju samo jednom:", len(jednom))
    print("Riječi koje se pojavljuju samo jednom:")
    print(jednom)

except FileNotFoundError:
    print("Datoteka 'song.txt' ne postoji.")