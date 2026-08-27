' QQ 2D Billiard Aim - launcher (ASCII only, CRLF, no BOM)
' Runs start.bat via cmd. Bypasses broken .bat double-click association.
' Writes vbs_trace.log BEFORE launching, for next-round diagnostics.
On Error Resume Next
Dim sh, fso, f, errMsg
errMsg = ""
Set sh = CreateObject("WScript.Shell")
If Err.Number <> 0 Then errMsg = "Create WScript.Shell: " & Err.Description
Err.Clear
Set fso = CreateObject("Scripting.FileSystemObject")
If Err.Number <> 0 Then errMsg = errMsg & vbCrLf & "Create FSO: " & Err.Description
Err.Clear
If Not fso Is Nothing Then
    Set f = fso.OpenTextFile("D:\QQGame\snoke\vbs_trace.log", 8, True)
    If Err.Number = 0 Then
        f.WriteLine "VBS RAN " & Year(Now) & "-" & Month(Now) & "-" & Day(Now) & " " & Hour(Now) & ":" & Minute(Now)
        f.Close
    End If
    Err.Clear
End If
If sh Is Nothing Then
    MsgBox "Launcher failed:" & vbCrLf & errMsg, 16, "QQ Billiard Aim"
    WScript.Quit 1
End If
On Error GoTo 0
sh.Run "cmd /c D:\QQGame\snoke\start.bat", 1, False
