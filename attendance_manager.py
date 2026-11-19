
import csv
from pathlib import Path
from datetime import datetime


class AttendanceManager:

    def __init__(self, attendance_file="data/attendance.csv"):
        self.attendance_file = Path(attendance_file)
        self.attendance_file.parent.mkdir(parents=True, exist_ok=True)

        self.fields = [
            "name",
            "roll_number",
            "date",
            "time"
        ]

        self._initialize_file()

    def _initialize_file(self):
        if not self.attendance_file.exists():
            with open(self.attendance_file, "w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=self.fields)
                writer.writeheader()

    def mark_attendance(self, name, roll_number):
        now = datetime.now()

        current_date = now.strftime("%Y-%m-%d")
        current_time = now.strftime("%H:%M:%S")

        # Prevent duplicate attendance for the same student on the same day
        with open(self.attendance_file, "r", newline="") as file:
            reader = csv.DictReader(file)

            for record in reader:
                if (
                    record["roll_number"] == roll_number
                    and record["date"] == current_date
                ):
                    return {
                        "success": False,
                        "message": "Attendance already marked today."
                    }

        # Save the attendance record
        with open(self.attendance_file, "a", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=self.fields)

            writer.writerow({
                "name": name,
                "roll_number": roll_number,
                "date": current_date,
                "time": current_time
            })

        return {
            "success": True,
            "message": "Attendance marked successfully.",
            "name": name,
            "roll_number": roll_number,
            "date": current_date,
            "time": current_time
        }


if __name__ == "__main__":
    manager = AttendanceManager()

    result = manager.mark_attendance(
        name="akshat",
        roll_number="24uec272"
    )

    print(result)
