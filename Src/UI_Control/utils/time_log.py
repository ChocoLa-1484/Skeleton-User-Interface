"""Record analysis elapsed time into Results/time.xlsx."""
from utils.paths import RECORD_ROOT

TIME_XLSX = RECORD_ROOT / 'time.xlsx'


def format_minutes(seconds):
    """Match the sheet's existing style: minutes.seconds.centiseconds (e.g. 50.17.14)."""
    centis = int(round(max(0.0, seconds) * 100))
    minutes, centis = divmod(centis, 6000)
    secs, centis = divmod(centis, 100)
    return f'{minutes}.{secs:02d}.{centis:02d}'


def record_time(name, seconds, path=TIME_XLSX):
    """Write the time into the row whose `name` matches; append a new row if none.

    Returns the text written. Raises PermissionError if the file is open in Excel.
    """
    from openpyxl import Workbook, load_workbook  # lazy: app still starts without openpyxl

    text = format_minutes(seconds)
    if path.exists():
        workbook = load_workbook(path)
        sheet = workbook.worksheets[0]
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        workbook = Workbook()
        sheet = workbook.active
        sheet.append(['name', 'time'])
    for row in range(2, sheet.max_row + 1):
        if sheet.cell(row, 1).value == name:
            sheet.cell(row, 2).value = text
            break
    else:
        sheet.append([name, text])
    workbook.save(path)
    return text
