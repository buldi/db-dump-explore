#!/bin/bash

# Ten skrypt automatyzuje proces aktualizacji plików lokalizacyjnych.
#
# Użycie:
#   ./update_locales.sh        - Aktualizuje wszystkie istniejące tłumaczenia.
#   ./update_locales.sh <kod>  - Tworzy nowe tłumaczenie dla danego kodu języka (np. 'de', 'fr').

# --- Konfiguracja ---
set -e # Przerwij działanie skryptu w przypadku błędu

APP_NAME="optimize_sql_dump"
SOURCE_FILE="optimize_sql_dump.py"
LOCALE_DIR="locale"
POT_FILE="${LOCALE_DIR}/${APP_NAME}.pot"

# --- Funkcje ---

update_pot_file() {
    echo "1. Aktualizowanie pliku szablonu '${POT_FILE}' z '${SOURCE_FILE}'..."
    # Używamy opcji, aby upewnić się, że plik .pot jest generowany z nagłówkiem UTF-8
    xgettext --keyword=_ --from-code=UTF-8 -d "${APP_NAME}" -o "${POT_FILE}" \
        --add-comments="TRANSLATORS:" -L Python \
        --package-name="${APP_NAME}" --msgid-bugs-address="dobrakowskirafal@gmail.com" "${SOURCE_FILE}"

    # Wymuś kodowanie UTF-8 w nagłówku pliku .pot, aby uniknąć błędów
    sed -i 's/charset=CHARSET/charset=UTF-8/' "${POT_FILE}"
    echo "   ...Gotowe."
    echo ""
}

update_and_compile_po_files() {
    echo "--- Rozpoczynanie aktualizacji istniejących lokalizacji ---"
    update_pot_file

    echo "2. Wyszukiwanie plików .po do aktualizacji i kompilacji..."

    if ! find "${LOCALE_DIR}" -name "*.po" -print -quit | grep -q .; then
        echo "   Nie znaleziono żadnych plików .po. Użyj skryptu z kodem języka, aby utworzyć nowy, np. './update_locales.sh pl'."
        echo "--- Aktualizacja lokalizacji zakończona (bez zmian) ---"
        return
    fi

    for po_file in $(find "${LOCALE_DIR}" -name "*.po"); do
        echo "   -> Znaleziono plik: ${po_file}"
        echo "      - Scalanie nowych tekstów..."
        msgmerge --update "${po_file}" "${POT_FILE}"

        mo_file="${po_file%.po}.mo"
        echo "      - Kompilowanie do pliku ${mo_file}..."
        msgfmt -o "${mo_file}" "${po_file}"
        echo "      ...Gotowe."
    done
    echo ""
    echo "--- Aktualizacja lokalizacji zakończona pomyślnie ---"
}

create_new_language() {
    LANG_CODE=$1
    echo "--- Próba utworzenia nowego tłumaczenia dla języka: ${LANG_CODE} ---"

    PO_FILE="${LOCALE_DIR}/${LANG_CODE}/LC_MESSAGES/${APP_NAME}.po"

    if [ -f "${PO_FILE}" ]; then
        echo "Błąd: Plik tłumaczenia '${PO_FILE}' już istnieje."
        echo "Aby go zaktualizować, uruchom skrypt bez argumentów."
        exit 1
    fi

    update_pot_file

    echo "2. Tworzenie nowego pliku .po dla języka '${LANG_CODE}'..."
    # Upewnij się, że katalog docelowy istnieje
    mkdir -p "$(dirname "${PO_FILE}")"

    msginit --input="${POT_FILE}" --output-file="${PO_FILE}" --locale="${LANG_CODE}"
    echo "   ...Gotowe."
    echo ""
    echo "Utworzono nowy plik tłumaczenia: ${PO_FILE}"
    echo "Teraz możesz go edytować i uzupełnić tłumaczenia."
    echo "--- Tworzenie nowego tłumaczenia zakończone pomyślnie ---"
}

# --- Główna logika skryptu ---
if [ -n "$1" ]; then
    create_new_language "$1"
else
    update_and_compile_po_files
fi
