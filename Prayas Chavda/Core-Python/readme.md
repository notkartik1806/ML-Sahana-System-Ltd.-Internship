# Student Management System

A simple console-based Student Management System built with Python for managing student records efficiently.

## Features

- **Add Student** - Add a single student with name, age, and marks
- **Bulk Add Students** - Add multiple students at once
  - Manual entry mode
  - Import from CSV file
- **View Students** - Display all student records
- **Search Student** - Find a student by name
- **Delete Student** - Remove a student record

## Requirements

- Python 3.x

## Usage

Run the program:
```bash
python main.py
```

### Menu Options

1. **Add a student** - Enter details for one student
2. **Bulk add students** - Choose between:
   - Manual entry: Specify count and enter details for multiple students
3. **View student list** - Display all students
4. **Search for a student** - Search by student name
5. **Delete a student** - Remove a student by name
6. **Exit** - Close the application

## File Structure

```
Core-Python/
├── main.py              # Main program with menu interface
├── student.py           # Student classes (Add, View, Search, Delete, BulkAdd)
├── data.txt             # Student data storage
```

## Example

```
========================================
   STUDENT MANAGEMENT SYSTEM
========================================
1. Add a student
2. Bulk add students
3. View student list
4. Search for a student
5. Delete a student
6. Exit
========================================
Enter your choice (1-6): 2

--- Bulk Add Students ---
Add multiple students manually
...
...
.
.
.
...
Successfully added 6 student(s)!
```

## Functions Overview

| Class | Purpose |
|-------|---------|
| `AddStudent` | Add a single student record |
| `BulkAddStudent` | Add multiple students |
| `ViewStudent` | Display all student records |
| `SearchStudent` | Search for a specific student |
| `DeleteStudent` | Remove a student record |

## Data Storage

Student data is stored in `data.txt` in CSV format:
```
Name,Age,Marks
```

---

**Author**: Prayas Chavda  
**Year**: 2026
