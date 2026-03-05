def total_euro(radni_sati, eura_po_satu):
    return radni_sati * eura_po_satu

radni_sati = float(input("Radni sati: "))
eura_po_satu = float(input("eura/h: "))

ukupno = total_euro(radni_sati, eura_po_satu)

print("Ukupno:", ukupno, "eura")