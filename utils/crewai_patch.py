"""
Patch for typing.Self in Python 3.10.
"""

import sys
import typing

# Only apply the patch if Python version is 3.10
if sys.version_info.major == 3 and sys.version_info.minor == 10:
    try:
        # Check if Self is already defined
        typing.Self
    except AttributeError:
        # Import Self from typing_extensions and add it to typing module
        from typing_extensions import Self
        typing.Self = Self
        
        print("Applied typing.Self patch for Python 3.10") 