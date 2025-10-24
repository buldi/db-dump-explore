#!/bin/bash

# This script automates the process of updating localization files.
#
# Usage:
#   ./update_locales.sh        - Updates all existing translations.
#   ./update_locales.sh <code>  - Creates a new translation for the given language code (e.g., 'de', 'fr').

# --- Configuration ---
set -e # Exit immediately if a command exits with a non-zero status.

APP_NAME="optimize_sql_dump"
SOURCE_FILE="optimize_sql_dump.py"
LOCALE_DIR="locale"
POT_FILE="${LOCALE_DIR}/${APP_NAME}.pot"

# --- Functions ---

update_pot_file() {
	echo "1. Updating template file '${POT_FILE}' from '${SOURCE_FILE}'..."
	# Use options to ensure the .pot file is generated with a UTF-8 header
	xgettext --keyword=_ --from-code=UTF-8 -d "${APP_NAME}" -o "${POT_FILE}" \
		--add-comments="TRANSLATORS:" -L Python \
		--package-name="${APP_NAME}" --msgid-bugs-address="dobrakowskirafal@gmail.com" "${SOURCE_FILE}"

	# Force UTF-8 encoding in the .pot file header to avoid errors
	sed -i 's/charset=CHARSET/charset=UTF-8/' "${POT_FILE}"
	echo "   ...Done."
	echo ""
}

update_and_compile_po_files() {
	echo "--- Starting update of existing localizations ---"
	update_pot_file

	echo "2. Searching for .po files to update and compile..."

	if ! find "${LOCALE_DIR}" -name "*.po" -print -quit | grep -q .; then
		echo "   No .po files found. Use the script with a language code to create a new one, e.g., './update_locales.sh en'."
		echo "--- Localization update finished (no changes) ---"
		return
	fi

	for po_file in $(find "${LOCALE_DIR}" -name "*.po"); do
		echo "   -> Found file: ${po_file}"
		echo "      - Merging new strings..."
		msgmerge --update "${po_file}" "${POT_FILE}"

		mo_file="${po_file%.po}.mo"
		echo "      - Compiling to ${mo_file}..."
		msgfmt -o "${mo_file}" "${po_file}"
		echo "      ...Done."
	done
	echo ""
	echo "--- Localization update completed successfully ---"
}

create_new_language() {
	LANG_CODE=$1
	echo "--- Attempting to create a new translation for language: ${LANG_CODE} ---"

	PO_FILE="${LOCALE_DIR}/${LANG_CODE}/LC_MESSAGES/${APP_NAME}.po"

	if [ -f "${PO_FILE}" ]; then
		echo "Error: Translation file '${PO_FILE}' already exists."
		echo "To update it, run the script without arguments."
		exit 1
	fi

	update_pot_file

	echo "2. Creating new .po file for language '${LANG_CODE}'..."
	# Ensure the target directory exists
	mkdir -p "$(dirname "${PO_FILE}")"

	msginit --input="${POT_FILE}" --output-file="${PO_FILE}" --locale="${LANG_CODE}"
	echo "   ...Done."
	echo ""
	echo "New translation file created: ${PO_FILE}"
	echo "You can now edit it and fill in the translations."
	echo "--- New translation creation completed successfully ---"
}

# --- Main script logic ---
if [ -n "$1" ]; then
	create_new_language "$1"
else
	update_and_compile_po_files
fi
