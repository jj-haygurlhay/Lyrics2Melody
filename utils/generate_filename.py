import datetime

def generate_filename(base, extension):
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    return f"{base}_{timestamp}.{extension}"