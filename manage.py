"""Django's command-line utility for administrative tasks."""
import os
import sys
import django

def main():
    """Run administrative tasks."""
    
    # Set the Django settings module
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'brandmonitor.settings')
    
    # Initialize Django settings
    django.setup()

    # Now it's safe to import Django-related modules
    from monitor.views import print_env_variables

    PYTHON_ENVIRONMENT = os.getenv('PYTHON_ENVIRONMENT', 'development')
    if PYTHON_ENVIRONMENT == 'development':
        print_env_variables()

    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()
