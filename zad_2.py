try:
    unos = input("Unesite ocjenu (0.0 - 1.0): ")
    ocjena = float(unos)

    if ocjena < 0.0 or ocjena > 1.0:
        print("Greška: broj mora biti u intervalu [0.0, 1.0].")
    else:
        match ocjena:
            case x if x >= 0.9:
                print("A")
            case x if x >= 0.8:
                print("B")
            case x if x >= 0.7:
                print("C")
            case x if x >= 0.6:
                print("D")
            case _:
                print("F")

except ValueError:
    print("Greška: niste unijeli broj.")