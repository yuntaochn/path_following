%%%===================================================%%%
%%%                This is actafa.cls                 %%%
%%%           For Control and Decision                %%%
%%%          Copyright by http://www.latexstudio.net  %%%
%%%  若有排版需要可以访问：http://www.latexstudio.net %%%
%%%===================================================%%%
%%%  revised by LaTeXStudio on 11/03/2016

%% This class is based on the class "ctexart" with options a4paper, oneside and 12pt.
%% The draft mode is defined, in which the package refcheck will take effect.
%% No more options are allowed.

\NeedsTeXFormat{LaTeX2e}[2012/09/01]
\ProvidesClass{ctacn}
\newif\if@cnpaper%
\@cnpapertrue%
\newif\if@authordraft%
\@authordrafttrue%
\LoadClass[twoside,c5size,UTF8]{ctexart}
\DeclareOption{enpaper}{\@cnpaperfalse}
\DeclareOption{cnpaper}{\@cnpapertrue}
\DeclareOption{authordraft}{\@authordraftfalse}
\ProcessOptions
\RequirePackage{graphicx}
\RequirePackage{amsmath,amsthm,bm,multicol}
\RequirePackage{amssymb,booktabs,longtable,lscape}
\RequirePackage[numbers,sort&compress,square,super]{natbib}
\RequirePackage{ifxetex,times,wasysym}
\usepackage{setspace}
\usepackage{upgreek}
\usepackage{multirow}
\ifxetex
\setmainfont{Times New Roman}
\fi
%---------page layout-----------
\RequirePackage{geometry}
\geometry{a4paper,text={170mm,257mm},top=2.3cm,headsep=2mm,headheight=1cm,hcentering} %修改版面距离，
\setlength{\columnsep}{2em}%
\setlength{\arraycolsep}{2.5pt}%
 

%------------abstract------------
\renewcommand{\maketitle}{\par
 \begingroup
    \setcounter{footnote}{-1}%
   \def\thefootnote{\fnsymbol{footnote}}%
   \long\def\@makefntext##1{\parindent 1em\noindent\hbox to 0pt{\hss$^{}$}##1}%
   \if@twocolumn
     \twocolumn[\if@cnpaper\@maketitle\else\@makeentitle\fi]%
     \thispagestyle{actasheadings}%ssl
   \else
     \if@cnpaper\@maketitle\else\@makeentitle\fi%
     \thispagestyle{actasheadings}%ssl
   \fi%
 \endgroup%
 \setcounter{footnote}{-1}%
}%
\renewenvironment{quote}
               {\list{}{\leftmargin=2.5em\rightmargin=2.5em}%
                \item\relax}
               {\endlist}
  
\def\clc#1{\gdef\cnactclc{{\noindent\zihao{-5}{\heiti \zhtu}{\bf: }#1}}}
\def\wenxianbiaoshi#1{\gdef\cnwenxian{{\noindent\zihao{-5}{\heiti \wxbs\bf: }#1}}}
 
\newcommand{\cntitle}[1]{\gdef\cta@cntitle{\LARGE\zihao{-2}\heiti#1}}
%\newcommand{\cnauthor}[1]{\gdef\cta@cnauthor{#1}}
\gdef\@cnauthor{}
\newcommand*\cnauthor[2]{%
  \ifx \@author\@empty%
    \gdef\@cnauthor{\@cnauthorformat{#1}{#2}}
  \else
    \expandafter\gdef\expandafter\@cnauthor\expandafter
      {\@cnauthor\@cnauthorformat{#1}{#2}}
  \fi}
\def\@cnauthorformat#1#2{
  \centering
  \vskip3.8mm%
    \cnauthorformat #1\\%
	\vskip2mm%
    \cnaddressformat 
	\begin{tabular}{@{}l@{}}
	#2\\%
	\end{tabular}%
  \vskip2.6mm%
}
\newcommand*{\cnaddress}[1]{\gdef\caddress{#1}}

\def\cnauthorformat{\centering\kaishu\zihao{-4}}
\def\cnaddressformat{\scriptsize\songti\zihao{-5}}

\newcommand*\Volume[5]{%
  \def\actas@year{#1}%
  \def\cnactas@monht{#2}%
  \def\actas@month{#3}%
  \def\actas@volume{#4}%
  \def\actas@number{#5}}

\def\titlefootnote#1{
   \begingroup
   \long\def\@makefntext[1]%
   {\noindent\makebox[0pt][r]{\@thefnmark\,}#1}
        \footnotetext{#1}%
   \endgroup}
\renewcommand\footnoterule{%
  \vskip .3em%
  \kern-3\p@
  \hrule\@width.4\columnwidth
  \kern2.6\p@}
\def\@maketitle{%
    \null\vskip-1.4em
	%\if@cnpaper%
	%{\papernum\hfill%
	%\ \quad\qquad}%
	%\fi
    \footnotesize%
    \let\footnote\titlefootnote%
	\def\thefootnote{\fnsymbol{footnote}}%
	\setcounter{footnote}{-1}%
    \vskip2.2mm%
    {\centering%
    {\begin{spacing}{1.3}\cta@cntitle\end{spacing}}%
    \vspace*{-1.7em}%
	\if@authordraft%
	{%
    \@cnauthor%
    \footnotetext{\vskip-1.3em\begin{spacing}{1.2}\parskip.8mm\accept \mod\par \ifx\ejijin\empty\else\jijin\par\fi \ifx\ebianwei\empty\else\bianwei\par\fi\autongxun\vspace*{-2.5em}\end{spacing}}}%
    \vspace*{-2mm}%
	\else%
	\footnotetext{\vskip-1.3em\begin{spacing}{1.2}\parskip.8mm\accept \mod\par \ifx\ejijin\empty\else\jijin\par\fi \ifx\ebianwei\empty\else\bianwei\par\fi\autongxun\vspace*{-2.5em}\end{spacing}}
	\fi
	{\begin{quote}\linespread{1.65}\selectfont%摘要行距
	\cabstract\newline\hbox{}%
    \ckeyword\par%
	\cnactclc  \quad\quad\quad\quad\quad  \cnwenxian\par
    \endoi\hfill\ifx\osidp\empty\else\osidtext   \setlength{\unitlength}{1cm}
\begin{picture}(1.5, 0)
    \includegraphics[height=15mm,width=15mm]{\osidp}
    \end{picture}\fi\par%
	\end{quote}%
    }%
   \@makeentitle}
    \vspace*{-1.95em}%
}

\newcommand*{\entitle}[1]{\gdef\cta@entitle{{\zihao{4}\bf #1}}}
\newcommand*{\enaddress}[1]{\gdef\cta@enaddress{{\noindent\it #1}}}
\newcommand*{\enabstract}[1]{\gdef\cta@enabstract{{\zihao{-5}\noindent{\bf Abstract:}  #1}}}
\newcommand*{\enkeyword}[1]{\gdef\cta@enkeyword{{\zihao{-5}\noindent{\bf Keywords:} #1}}}
\gdef\ensub{\noindent{\bf MR(2000) Subject Classification:} }

\def\enauthorformat{\normalsize\itshape}
\def\enaddressformat{\normalfont\zihao{6}}


\gdef\@enauthor{}
\newcommand*\enauthor[2]{%
  \ifx \@enauthor\@empty%
    \gdef\@enauthor{\@enauthorformat{#1}{#2}}
  \else
    \expandafter\gdef\expandafter\@enauthor\expandafter
      {\@enauthor\@enauthorformat{#1}{#2}}
  \fi}
\def\@enauthorformat#1#2{
  %\begin{center}
    \enauthorformat #1\par%
	\vskip 0.2\baselineskip%
    \enaddressformat%
	\begin{spacing}{1.3}\zihao{-5}#2\end{spacing}
  %\end{center}
  }


\def\@makeentitle{\vspace*{-1em}%修改英文标题以上距离
    \begingroup%
	\null%an empty hbox
    \vskip .1\baselineskip%
    \begin{quote}{%
    {{\begin{spacing}{2.0}\cta@entitle\end{spacing}}\par}%
    \if@authordraft%
	{\vskip -0.5em%
    \vskip 0.2\baselineskip%
    \@enauthor\par%
    }%
	\else
	\fi
	}%
    %\vskip2mm%
    \begin{spacing}{1.5}\cta@enabstract\newline\hbox{}%
    \cta@enkeyword%
	\end{spacing}\end{quote}%
	\endgroup
}% 

% \def\@hangfrom#1{\setbox\@tempboxb\hbox{{#1}}%
  % \hangindent \wd\@tempboxa\noindent\box\@tempboxa}
 \newbox\@tempboxb
\long\def\@makecaption#1#2{%
  %\vskip\abovecaptionskip
  \sbox\@tempboxa{#1  #2}%
  \setbox\@tempboxb\hbox{{#1 }}%
 \ifdim \wd\@tempboxa >\hsize
  \hangindent\wd\@tempboxb\noindent\box\@tempboxb\heiti#2\par
  \else
    \global \@minipagefalse
    \hb@xt@\hsize{\hfil\box\@tempboxa\hfil}%
  \fi
  \vskip\belowcaptionskip}
\newcounter{fignum}
\def\figcaption#1{
\refstepcounter{fignum}%
{\@makecaption{\heiti\zihao{-5}\figurename~{\bfseries\arabic{fignum}}~~\ }{\zihao{-5}#1}}
%\vskip-10pt%
%\@makecaption{\zihao{-5}\rm Fig. \arabic{fignum}}{\zihao{-5} #2}}%
}
\newcounter{tabnum}
\def\tabcaption#1{
\refstepcounter{tabnum}%
{\@makecaption{\heiti\zihao{-5}  \tablename~{\bfseries\arabic{tabnum}}~~\ }{\zihao{-5} #1}}
\vskip5pt\zihao{6}
%\vskip-10pt%
%\@makecaption{\zihao{-5}\rm  Tab~\arabic{tabnum}}{\zihao{-5}#2}}%
}

%%---------------- Redefine section -----------------%%
\setcounter{secnumdepth}{3}
 
\renewcommand\thesubsection   {\bf\thesection.\@arabic\c@subsection}
\renewcommand\thesubsubsection{\thesubsection .\@arabic\c@subsubsection}
\renewcommand\section{\@startsection {section}{1}{\z@}%
                                     {0.7ex}%
                                     {0.3ex}%
                                     {\normalfont\zihao{-4}\bfseries}}
\renewcommand\subsection{\@startsection{subsection}{2}{\z@}%
                                     {0.5ex}%
                                     {0.3ex}%
                                     {\normalfont\zihao{5}\heiti}}
\renewcommand\subsubsection{\@startsection{subsubsection}{3}{\z@}%
                                     {0.4ex}%
                                     {0.2ex}%
                                     {\normalfont\zihao{5}\bfseries}}
\renewcommand\paragraph{\@startsection{paragraph}{4}{\z@}%
                                     {3.25ex \@plus1ex \@minus.2ex}%
                                     {-1em}%
                                     {\normalfont\normalsize\itshape}}
\renewcommand\subparagraph{\@startsection{subparagraph}{5}{\parindent}%
                                     {3.25ex \@plus1ex \@minus .2ex}%
                                     {-1em}%
                                     {\normalfont\normalsize\itshape}}
\def\@seccntformat#1{\bf\csname the#1\endcsname\quad}
%字号

\def\CTEX@fs@eight{5.02}   \def\CTEX@fs@eightskip{6.02}   %�˺���    5bp
\def\CTEX@fs@seven{5.52}   \def\CTEX@fs@sevenskip{6.62}   %�ߺ���  5.5bp
\def\CTEX@fs@ssix{6.92}    \def\CTEX@fs@ssixskip{7.83}    %С����  6.5bp
\def\CTEX@fs@six{7.93}     \def\CTEX@fs@sixskip{9.03}     %������  7.5bp
\def\CTEX@fs@sfive{9.23}   \def\CTEX@fs@sfiveskip{10.84}  %С����    9bp
\def\CTEX@fs@five{10.34}   \def\CTEX@fs@fiveskip{12.65}   %������ 10.5bp
\def\CTEX@fs@sfour{12.25}  \def\CTEX@fs@sfourskip{14.45}  %С�ĺ�   12bp
\def\CTEX@fs@four{14.25}   \def\CTEX@fs@fourskip{16.86}   %�ĺ���   14bp
\def\CTEX@fs@sthree{15.26} \def\CTEX@fs@sthreeskip{18.07} %С����   15bp
\def\CTEX@fs@three{16.26}  \def\CTEX@fs@threeskip{19.27}  %������   16bp
\def\CTEX@fs@stwo{17.87}   \def\CTEX@fs@stwoskip{21.68}   %С����   18bp
\def\CTEX@fs@two{22.28}    \def\CTEX@fs@twoskip{26.50}    %������   22bp
\def\CTEX@fs@sone{23.89}   \def\CTEX@fs@soneskip{28.91}   %Сһ��   24bp
\def\CTEX@fs@one{25.90}    \def\CTEX@fs@oneskip{31.32}    %һ����   26bp
\def\CTEX@fs@szero{36.14}  \def\CTEX@fs@szeroskip{43.36}  %С����   36bp
\def\CTEX@fs@zero{42.16}   \def\CTEX@fs@zeroskip{50.59}   %������   42bp
\DeclareMathSizes{\CTEX@fs@eight}{\CTEX@fs@eight}{5}{5}
\DeclareMathSizes{\CTEX@fs@seven}{\CTEX@fs@seven}{5}{5}
\DeclareMathSizes{\CTEX@fs@ssix}{\CTEX@fs@ssix}{5}{5}
\DeclareMathSizes{\CTEX@fs@six}{\CTEX@fs@six}{5}{5}
\DeclareMathSizes{\CTEX@fs@sfive}{\CTEX@fs@sfive}{6}{5}
\DeclareMathSizes{\CTEX@fs@five}{\CTEX@fs@five}{7}{5}
\DeclareMathSizes{\CTEX@fs@sfour}{\CTEX@fs@sfour}{8}{6}
\DeclareMathSizes{\CTEX@fs@four}
                 {\CTEX@fs@four}{\CTEX@fs@five}{\CTEX@fs@six}
\DeclareMathSizes{\CTEX@fs@sthree}
                 {\CTEX@fs@sthree}{\CTEX@fs@sfour}{\CTEX@fs@sfive}
\DeclareMathSizes{\CTEX@fs@three}
                 {\CTEX@fs@three}{\CTEX@fs@four}{\CTEX@fs@five}
\DeclareMathSizes{\CTEX@fs@stwo}
                 {\CTEX@fs@stwo}{\CTEX@fs@sthree}{\CTEX@fs@sfour}
\DeclareMathSizes{\CTEX@fs@two}
                 {\CTEX@fs@two}{\CTEX@fs@three}{\CTEX@fs@four}
\DeclareMathSizes{\CTEX@fs@sone}
                 {\CTEX@fs@sone}{\CTEX@fs@stwo}{\CTEX@fs@sthree}
\DeclareMathSizes{\CTEX@fs@one}
                 {\CTEX@fs@one}{\CTEX@fs@two}{\CTEX@fs@three}
\DeclareMathSizes{\CTEX@fs@szero}
                 {\CTEX@fs@szero}{\CTEX@fs@sone}{\CTEX@fs@stwo}
\DeclareMathSizes{\CTEX@fs@zero}
                 {\CTEX@fs@zero}{\CTEX@fs@one}{\CTEX@fs@two}
\def\CTEX@zihao{}
\DeclareRobustCommand*\zihao[1]{\def\CTEX@zihao{#1}%
  \ifnum #11<0%
    \@tempcnta=-#1
    \ifcase\@tempcnta%
        \fontsize\CTEX@fs@szero\CTEX@fs@szeroskip%
    \or \fontsize\CTEX@fs@sone\CTEX@fs@soneskip%
    \or \fontsize\CTEX@fs@stwo\CTEX@fs@stwoskip%
    \or \fontsize\CTEX@fs@sthree\CTEX@fs@sthreeskip%
    \or \fontsize\CTEX@fs@sfour\CTEX@fs@sfourskip%
    \or \fontsize\CTEX@fs@sfive\CTEX@fs@sfiveskip%
    \or \fontsize\CTEX@fs@ssix\CTEX@fs@ssixskip%
    \else \PackageError{ctex}{%
            Undefined Chinese font size in command \protect\zihao}{%
            The old font size is used if you continue.}%
    \fi%
  \else%
    \@tempcnta=#1
    \ifcase\@tempcnta%
        \fontsize\CTEX@fs@zero\CTEX@fs@zeroskip%
    \or \fontsize\CTEX@fs@one\CTEX@fs@oneskip%
    \or \fontsize\CTEX@fs@two\CTEX@fs@twoskip%
    \or \fontsize\CTEX@fs@three\CTEX@fs@threeskip%
    \or \fontsize\CTEX@fs@four\CTEX@fs@fourskip%
    \or \fontsize\CTEX@fs@five\CTEX@fs@fiveskip%
    \or \fontsize\CTEX@fs@six\CTEX@fs@sixskip%
    \or \fontsize\CTEX@fs@seven\CTEX@fs@sevenskip%
    \or \fontsize\CTEX@fs@eight\CTEX@fs@eightskip%
    \else \PackageError{ctex}{%
            Undefined Chinese font size in command \protect\zihao}{%
            The old font size is used if you continue.}%
    \fi%
  \fi%
  \selectfont\ignorespaces}

% font size definitions

  \renewcommand\normalsize{% 10.5bp
    \@setfontsize\normalsize{\CTEX@fs@five}{\CTEX@fs@fiveskip}%
    \abovedisplayskip 3\p@ \@plus2\p@ \@minus5\p@
    \abovedisplayshortskip \z@ \@plus3\p@
    \belowdisplayshortskip 3\p@ \@plus3\p@ \@minus3\p@
    \belowdisplayskip \abovedisplayskip
    \let\@listi\@listI}
  \normalsize
  \renewcommand\small{% 9bp
    \@setfontsize\small{\CTEX@fs@sfive}{\CTEX@fs@sfiveskip}%
    \abovedisplayskip 8.5\p@ \@plus3\p@ \@minus4\p@
    \abovedisplayshortskip \z@ \@plus2\p@
    \belowdisplayshortskip 4\p@ \@plus2\p@ \@minus2\p@
    \def\@listi{\leftmargin\leftmargini
                \topsep 4\p@ \@plus2\p@ \@minus2\p@
                \parsep 2\p@ \@plus\p@ \@minus\p@
                \itemsep \parsep}%
    \belowdisplayskip \abovedisplayskip}
  \renewcommand\footnotesize{% 7.5bp
    \@setfontsize\footnotesize{\CTEX@fs@six}{\CTEX@fs@sixskip}%
    \abovedisplayskip 6\p@ \@plus2\p@ \@minus4\p@
    \abovedisplayshortskip \z@ \@plus\p@
    \belowdisplayshortskip 3\p@ \@plus\p@ \@minus2\p@
    \def\@listi{\leftmargin\leftmargini
                \topsep 3\p@ \@plus\p@ \@minus\p@
                \parsep 2\p@ \@plus\p@ \@minus\p@
                \itemsep \parsep}%
    \belowdisplayskip \abovedisplayskip}
  \renewcommand\scriptsize{% 6.5bp
    \@setfontsize\scriptsize{\CTEX@fs@ssix}{\CTEX@fs@ssixskip}}
  \renewcommand\tiny{% 5.5bp
    \@setfontsize\tiny{\CTEX@fs@seven}{\CTEX@fs@sevenskip}}
  \renewcommand\large{% 12bp
    \@setfontsize\large{\CTEX@fs@sfour}{\CTEX@fs@sfourskip}}
  \renewcommand\Large{% 15bp
    \@setfontsize\Large{\CTEX@fs@sthree}{\CTEX@fs@sthreeskip}}
  \renewcommand\LARGE{% 18bp
    \@setfontsize\LARGE{\CTEX@fs@stwo}{\CTEX@fs@stwoskip}}
  \renewcommand\huge{% 22bp
    \@setfontsize\huge{\CTEX@fs@two}{\CTEX@fs@twoskip}}
  \renewcommand\Huge{% 26bp
    \@setfontsize\Huge{\CTEX@fs@one}{\CTEX@fs@oneskip}}

%  删除列表项之间的间隔.
\def\@listi{%
  \leftmargin\leftmargini
  \parsep 1.4mm%
  \topsep \parsep
  \itemsep \parsep}
\let\@listI\@listi
\@listi
\def\@listii{%
  \leftmargin\leftmarginii
  \labelwidth\leftmarginii
  \advance\labelwidth-\labelsep
  \parsep 0pt%
  \topsep \parsep
  \itemsep \parsep}
\def\@listiii{%
  \leftmargin\leftmarginiii
  \labelwidth\leftmarginiii
  \advance\labelwidth-\labelsep
  \parsep 0pt%
  \topsep \parsep
  \itemsep \parsep
  \partopsep \p@ \@plus\z@ \@minus\p@}
  
\setcounter{section}{-1}%
\renewenvironment{proof}[1][\proofname]{\par
  \pushQED{\qed}%
  \normalfont \topsep6\p@\@plus6\p@\relax
  \trivlist
  \item[\hskip\labelsep
    #1]\ignorespaces
}{%
  \popQED\endtrivlist\@endpefalse
}
\providecommand{\proofname}{Proof}
  \newenvironment{authorinfo}{%
        \leftline{\heiti\zihao{5}作者简介}%
      \zihao{5}}
      {}

\def\ps@headings{
\let\@oddfoot\@empty\let\@evenfoot\@empty%
 \def\@evenhead{\vbox{\hbox to \textwidth{\small\zihao{5}\fangsong\thepage\hfill  \kaishu\journal \hfill\fangsong\cndi\actas@volume\cnvol}
                \vskip 2mm%
               \hrule width\textwidth height0.63pt\vskip0.85pt}}
 \def\@oddhead{\vbox{\hbox to \textwidth{\small\rm\zihao{5}\fangsong \cndi\actas@number\cnnum  \hfill\kaishu\runhead \hfill\thepage}
	              \vskip 2mm%
                  \hrule width\textwidth height0.63pt\vskip0.85pt}}
}


%%------------------the biblogr
\newdimen\bibindent
%\setlength\bibindent{1em} 
\renewcommand{\@biblabel}[1]{[#1]\hfill}
\renewenvironment{thebibliography}[1]
     {\section*{\reset@font\heiti\bibname}%
      \@mkboth{\MakeUppercase\bibname}{\MakeUppercase\bibname}
      \vskip 2mm
      \list{\@biblabel{\@arabic\c@enumiv}}%
           {\settowidth\labelwidth{\@biblabel{#1}}%
            \leftmargin\labelwidth
            \advance\leftmargin\labelsep
			\addtolength{\itemsep}{0.3em}
            \@openbib@code
            \usecounter{enumiv}%
            \let\p@enumiv\@empty
            \renewcommand\theenumiv{\@arabic\c@enumiv}}%
      \sloppy
      \clubpenalty4000
      \@clubpenalty \clubpenalty
      \widowpenalty4000%
      \sfcode`\.\@m%
	  \setlength{\parskip}{0pt}\zihao{-5}\hyphenpenalty=10000%
      \exhyphenpenalty=10000}
     {\def\@noitemerr%
       {\@latex@warning{Empty `thebibliography' environment}}%
      \endlist}


\newcommand{\rb}[1]{\raisebox{2.6ex}[-2pt]{#1}}
%%---------keep  old  command for previous users ----------
\def\ke#1#2{\hbox to2cm{#1\hss}\parbox[t]{10.7cm}{#2}}
\def\SEC#1{\section*{\leftline{\large\bf\zihao{-4}\heiti #1}}}
\def\REF#1{\par\hangindent\parindent\indent\llap{#1\enspace}\ignorespaces}

\renewcommand\@makefntext[1]%
{\parindent 1em\indent\hbox to 0pt{\hss$^{}$}#1} 
%%-------- Set the begin and end method for class -----%%
\AtBeginDocument{%
   \pagestyle{headings}% 
   \input{ctacn.cfg}%
   \tolerance=1%
   \emergencystretch=\maxdimen
   \hyphenpenalty=10000
   \hbadness=10000
   \renewcommand{\baselinestretch}{1.2}
   \parskip0pt%
   }
%% End of file `actafa.cls'.  %long2014-7-22
