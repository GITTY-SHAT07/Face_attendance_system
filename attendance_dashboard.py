
import csv
from pathlib import Path
from datetime import datetime

ATTENDANCE_FILE = Path("data/attendance.csv")
EXPORT_DIR = Path("data/exports")


def load_attendance():
    if not ATTENDANCE_FILE.exists():
        print("Attendance file not found.")
        return []

    with open(ATTENDANCE_FILE, "r", newline="") as file:
        return list(csv.DictReader(file))


def display_records(records):
    if not records:
        print("\nNo attendance records found.")
        return

    print("\n" + "-" * 85)
    print(
        f"{'NAME':<20}"
        f"{'ROLL NUMBER':<20}"
        f"{'DATE':<18}"
        f"{'TIME':<15}"
    )
    print("-" * 85)

    for record in records:
        print(
            f"{record['name']:<20}"
            f"{record['roll_number']:<20}"
            f"{record['date']:<18}"
            f"{record['time']:<15}"
        )

    print("-" * 85)


def show_statistics(records):
    unique_students = {
        record["roll_number"] for record in records
    }

    print("\nATTENDANCE STATISTICS")
    print("-" * 35)
    print("Total attendance entries:", len(records))
    print("Unique students:", len(unique_students))

    if records:
        dates = sorted({record["date"] for record in records})
        print("First attendance date:", dates[0])
        print("Latest attendance date:", dates[-1])


def filter_by_date(records):
    selected_date = input(
        "\nEnter date (YYYY-MM-DD): "
    ).strip()

    try:
        datetime.strptime(selected_date, "%Y-%m-%d")
    except ValueError:
        print("Invalid date format.")
        return []

    filtered = [
        record for record in records
        if record["date"] == selected_date
    ]

    print(f"\nRecords for {selected_date}:")
    display_records(filtered)

    return filtered


def search_student(records):
    query = input(
        "\nEnter student name or roll number: "
    ).strip().lower()

    filtered = [
        record for record in records
        if query in record["name"].lower()
        or query in record["roll_number"].lower()
    ]

    display_records(filtered)

    return filtered


def export_records(records):
    if not records:
        print("No records available to export.")
        return

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    filename = input(
        "\nEnter export filename (without extension): "
    ).strip()

    if not filename:
        filename = "attendance_export"

    safe_filename = "".join(
        char for char in filename
        if char.isalnum() or char in ("-", "_")
    )

    if not safe_filename:
        print("Invalid filename.")
        return

    export_path = EXPORT_DIR / f"{safe_filename}.csv"

    fields = ["name", "roll_number", "date", "time"]

    with open(export_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    print("\nExport successful!")
    print("Saved to:", export_path)


def main():
    records = load_attendance()

    while True:
        print("\n========== ATTENDANCE DASHBOARD ==========")
        print("1. View all attendance")
        print("2. Filter attendance by date")
        print("3. Search student")
        print("4. View statistics")
        print("5. Export all attendance")
        print("6. Export attendance by date")
        print("7. Exit")

        choice = input("\nSelect an option: ").strip()

        if choice == "1":
            display_records(records)

        elif choice == "2":
            filter_by_date(records)

        elif choice == "3":
            search_student(records)

        elif choice == "4":
            show_statistics(records)

        elif choice == "5":
            export_records(records)

        elif choice == "6":
            filtered = filter_by_date(records)
            if filtered:
                export_records(filtered)

        elif choice == "7":
            print("Closing attendance dashboard.")
            break

        else:
            print("Invalid option. Please try again.")

        # Reload records in case attendance was added elsewhere.
        records = load_attendance()


if __name__ == "__main__":
    main()
