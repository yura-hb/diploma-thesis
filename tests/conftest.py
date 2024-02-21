import sys
import pytest


# each workflow training_schedules on cwd to its temp dir
@pytest.fixture(autouse=True)
def go_to_tmpdir(request):
    # Get the fixture dynamically by its name.
    tmpdir = request.getfixturevalue("tmpdir")
    # ensure local workflow created packages can be imported
    sys.path.insert(0, str(tmpdir))
    # Chdir only for the duration of the workflow.
    with tmpdir.as_cwd():
        yield
