Option Explicit

Dim shell, scriptDir, batchPath, command
Set shell = CreateObject("WScript.Shell")
scriptDir = Left(WScript.ScriptFullName, InStrRev(WScript.ScriptFullName, "\"))
batchPath = scriptDir & "start.bat"
command = Chr(34) & shell.ExpandEnvironmentStrings("%ComSpec%") & Chr(34) & _
    " /c " & Chr(34) & Chr(34) & batchPath & Chr(34) & Chr(34)

' 0 = hidden window; False = return immediately after starting the service.
shell.Run command, 0, False
