#ifndef AppVersion
  #define AppVersion "0.36.0"
#endif
#ifndef InstallerAppId
  #define InstallerAppId "{{7BC65B14-5BDF-4C8F-93E1-784B379E622D}"
#endif
#ifndef SourceDir
  #define SourceDir "..\.runtime\desktop-dist\VisionWorkbench"
#endif
#ifndef OutputDir
  #define OutputDir "..\deliverables"
#endif
[Setup]
AppId={#InstallerAppId}
AppName=视训工作台
AppVersion={#AppVersion}
AppPublisher=视训工作台
AppPublisherURL=https://github.com/studyhard-zhang/vision-workbench
DefaultDirName={localappdata}\Programs\VisionWorkbench
DefaultGroupName=视训工作台
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0.19041
OutputDir={#OutputDir}
OutputBaseFilename=VisionWorkbench-Setup-{#AppVersion}-x64
SetupIconFile=app.ico
UninstallDisplayIcon={app}\VisionWorkbench.exe
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
CloseApplications=yes
CloseApplicationsFilter=VisionWorkbench.exe
RestartApplications=no
SetupLogging=yes
[Languages]
Name: "chinesesimp"; MessagesFile: "ChineseSimplified.isl"
[Tasks]
Name: "desktopicon"; Description: "创建桌面快捷方式"; Flags: checkedonce
[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "使用说明.md"; DestDir: "{app}"; DestName: "使用说明.txt"; Flags: ignoreversion
Source: "..\.runtime\build-tools\MicrosoftEdgeWebview2Setup.exe"; Flags: dontcopy
[Icons]
Name: "{group}\视训工作台"; Filename: "{app}\VisionWorkbench.exe"
Name: "{userdesktop}\视训工作台"; Filename: "{app}\VisionWorkbench.exe"; Tasks: desktopicon
[Run]
Filename: "{app}\VisionWorkbench.exe"; Description: "启动视训工作台"; Flags: nowait postinstall skipifsilent
[Code]
var WorkspacePage: TInputDirWizardPage;
    ModePage: TInputOptionWizardPage;

function ProfileDir: String;
begin
  Result := ExpandConstant('{param:PROFILEDIR|{localappdata}\VisionWorkbench}');
end;

function HasWebView: Boolean;
var Version: String;
begin
  Result := RegQueryStringValue(HKLM32, 'SOFTWARE\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}', 'pv', Version) or
    RegQueryStringValue(HKCU, 'Software\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}', 'pv', Version);
  Result := Result and (Version <> '') and (Version <> '0.0.0.0');
end;

function InsideDir(ChildPath, ParentPath: String): Boolean;
begin
  Result := Pos(Lowercase(AddBackslash(ExpandFileName(ParentPath))),
    Lowercase(AddBackslash(ExpandFileName(ChildPath)))) = 1;
end;

function EmptyDirectory(Path: String): Boolean;
var Entry: TFindRec;
begin
  Result := True;
  if FindFirst(AddBackslash(Path) + '*', Entry) then begin
    try
      repeat
        if (Entry.Name <> '.') and (Entry.Name <> '..') then begin Result := False; break; end;
      until not FindNext(Entry);
    finally FindClose(Entry); end;
  end;
end;

function WorkspaceError: String;
var Workspace: String;
begin
  Result := '';
  Workspace := Trim(WorkspacePage.Values[0]);
  if (Length(Workspace) < 4) or FileExists(Workspace) then begin Result := '请选择有效的工作区目录。'; exit; end;
  Workspace := ExpandFileName(Workspace);
  if InsideDir(Workspace, WizardDirValue) or InsideDir(WizardDirValue, Workspace) or
     InsideDir(ProfileDir, WizardDirValue) then begin
    Result := '工作区和客户端配置须位于程序安装目录以外，且不能包含程序目录。'; exit;
  end;
  if ModePage.SelectedValueIndex = 0 then begin
    if DirExists(Workspace) and not EmptyDirectory(Workspace) then
      Result := '新工作区须为空目录。已有数据请选择“使用已有工作区”；不会覆盖或清空。';
  end else if not FileExists(AddBackslash(Workspace) + 'platform.db') then
    Result := '此目录没有 platform.db，请选择已有工作区。';
end;

procedure InitializeWizard;
var Previous, RequestedMode: String; Lines: TArrayOfString; I: Integer; HasPrevious: Boolean;
begin
  ModePage := CreateInputOptionPage(wpSelectDir, '选择工作区方式', '首次使用或继续已有数据',
    '工作区保存图片、标注、训练和模型，与程序分开。升级及卸载保留工作区。', True, False);
  ModePage.Add('首次使用：创建新工作区（空目录）');
  ModePage.Add('已有数据：使用已有工作区');
  WorkspacePage := CreateInputDirPage(ModePage.ID, '选择工作区目录', '选择新建空目录或已有 platform.db 所在目录',
    '新工作区从空项目列表开始。已有工作区会继续使用原数据，不导入演示项目或测试材料。', False, '');
  WorkspacePage.Add('工作区目录：');
  Previous := ExpandConstant('{localappdata}\VisionWorkbenchData\var');
  HasPrevious := False;
  if LoadStringsFromFile(ProfileDir + '\desktop.ini', Lines) then
    for I := 0 to GetArrayLength(Lines) - 1 do
      if Pos('directory=', Lines[I]) = 1 then begin Previous := Copy(Lines[I], 11, Length(Lines[I])); HasPrevious := True; end;
  RequestedMode := ExpandConstant('{param:WORKSPACEMODE|auto}');
  ModePage.SelectedValueIndex := 0;
  if (RequestedMode = 'existing') or ((RequestedMode = 'auto') and HasPrevious) then ModePage.SelectedValueIndex := 1;
  WorkspacePage.Values[0] := ExpandConstant('{param:WORKSPACE|' + Previous + '}');
end;

function NextButtonClick(CurPageID: Integer): Boolean;
var Problem: String;
begin
  Result := True;
  if CurPageID = WorkspacePage.ID then begin
    Problem := WorkspaceError;
    Result := Problem = '';
    if not Result then MsgBox(Problem, mbError, MB_OK);
  end;
end;

function PrepareToInstall(var NeedsRestart: Boolean): String;
var ResultCode: Integer;
begin
  Result := WorkspaceError;
  if Result <> '' then exit;
  if not HasWebView then begin
    ExtractTemporaryFile('MicrosoftEdgeWebview2Setup.exe');
    if not Exec(ExpandConstant('{tmp}\MicrosoftEdgeWebview2Setup.exe'), '/silent /install', '', SW_HIDE, ewWaitUntilTerminated, ResultCode) or not HasWebView then
      Result := 'WebView2运行时未就绪。首次安装此组件需要联网，请安装Microsoft WebView2后重试；工作区数据未改变。';
  end;
end;

procedure CurStepChanged(CurStep: TSetupStep);
var Lines: TArrayOfString; ResultCode: Integer; Arguments: String;
begin
  if CurStep = ssPostInstall then begin
    if ModePage.SelectedValueIndex = 0 then begin
      Arguments := '--initialize-workspace --data-dir "' + WorkspacePage.Values[0] + '" --profile-dir "' + ProfileDir + '"';
      if not Exec(ExpandConstant('{app}\VisionWorkbench.exe'), Arguments, '', SW_HIDE, ewWaitUntilTerminated, ResultCode) or (ResultCode <> 0) then
        RaiseException('工作区创建未完成。请查看客户端日志，选择空目录重试；已有数据未被替换。');
    end;
    ForceDirectories(ProfileDir);
    SetArrayLength(Lines, 2); Lines[0] := '[workspace]'; Lines[1] := 'directory=' + WorkspacePage.Values[0];
    if not SaveStringsToUTF8File(ProfileDir + '\desktop.ini', Lines, False) then
      RaiseException('无法保存工作区路径，请检查客户端配置目录权限。原业务数据未改变。');
  end;
end;
