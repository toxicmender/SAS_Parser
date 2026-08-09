## [meta: invokes_macros, defines_macros, produces_macrovars] [lang: pyspark] Macro processing

SAS macro expansion happens before execution and has no DataFrame equivalent.
Resolve known macro variables into Python constants or notebook parameters;
translate repeated macro invocations into Python functions only when their
inputs and generated operations are known. Flag unresolved dynamic code in the
risks instead of emitting a fake macro processor.
