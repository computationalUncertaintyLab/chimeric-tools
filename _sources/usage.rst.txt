Usage
=====

Installation
------------

To use chimeric_tools, first install it using pip:

.. code-block:: console

   $ pip install chimeric_tools


Prepocess 
--------------------------------

.. automodule:: chimeric_tools.Prepocess
   :members:
   :undoc-members:
   :show-inheritance:

Spatial 
------------------------------

.. .. automodule:: chimeric_tools.Spatial
..    :members:
..    :undoc-members:
..    :show-inheritance:
.. autofunction:: chimeric_tools.Spatial.GetStateAbbreviation(String)

>>> from chimeric_tools import Spatial
>>> Spatial.GetStateAbbreviation("Pennsylvania")
"PA"

.. autofunction:: chimeric_tools.Spatial.fromState2FIPS(String)

>>> from chimeric_tools import Spatial
>>> Spatial.fromState2FIPS("PA")
42

Time 
---------------------------
Contains some fuctions that can be used to show the epiweeks info.

.. .. automodule:: chimeric_tools.Time
..    :members:
..    :undoc-members:
..    :show-inheritance:

.. autofunction:: chimeric_tools.Time.todayEpiWeek

>>> from chimeric_tools import Time
>>> Time.todayEpiWeek()
12

.. autofunction:: chimeric_tools.Time.fromDate2Epiweek(String)

>>> from chimeric_tools import Time
>>> Time.fromDate2Epiweek(fromDate2Epiweek("2020-03-22"))
13

.. autofunction:: chimeric_tools.Time.fromDates2Epiweeks(List)

>>> from chimeric_tools import Time
>>> Time.fromDates2Epiweeks(["2020-01-22","2020-02-23","2020-03-24"])
[4, 9, 13]


.. Module contents
.. ---------------

.. .. automodule:: chimeric_tools
..    :members:
..    :undoc-members:
..    :show-inheritance:
