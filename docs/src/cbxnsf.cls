%=========================================================================
% NSF Proposal Class
%=========================================================================
% This class file formats a paper according to the NSF Proposal and
% Award Policies and Procedures Guide: Part I - Proposal Preparation and
% SUbmission Guidelines (GPG). In terms of formatting, we just need to
% use Times New Roman 11pt font, 1in margins, and 6 lines per inch.
%
% Class Options
%  - annotate   : show margin annotations
%  - draftcopy  : add draft copy & do not dist. header, draft watermark
%
% Author : Christopher Batten
% Date   : July 18, 2009
%

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{cbxnsf}[2009/07/18 ver 0.1]
\LoadClass[10pt]{article}

%-------------------------------------------------------------------------
% Options
%-------------------------------------------------------------------------
% These are the options which you can pass when you call \documentclass.

\gdef\cbxarticle@annotate{false}
\gdef\nsfproposal@draftcopy{false}

\DeclareOption{annotate}{\gdef\cbxarticle@annotate{true}}
\DeclareOption{draftcopy}{\gdef\nsfproposal@draftcopy{true}}
\ProcessOptions

%-------------------------------------------------------------------------
% Packages
%-------------------------------------------------------------------------
% We use the mathptmx package to load the Times fonts. The main text
% font size is set as an option when we load the article class. I also
% load packages which I pretty much always use (eg. graphicx, textcomp,
% booktabs, etc)

\RequirePackage{geometry}     % Page layout

\RequirePackage[T1]{fontenc}  % Make sure we use T1 font encoding
\RequirePackage{mathpazo}     % Use times for serif font
\RequirePackage[scaled=0.92]{helvet} % Use helvetica for sans serif font
%\RequirePackage{courier}      % Use courier for fixed width font

\RequirePackage{fixltx2e}      % Fix some basic latex issues
\RequirePackage{caption}      % updated version of caption
\RequirePackage{subcaption}   % Subfigure captions
\RequirePackage{graphicx}     % To include graphics
\RequirePackage{textcomp}     % For non-math characters (eg. mu)
\RequirePackage{cite}         % Sort cite lists [1,2,3]
\RequirePackage{fancyhdr}     % For headers and footers
\RequirePackage{booktabs}     % Very nice table formatting
\RequirePackage{xcolor}       % Colored text
\RequirePackage{url}          % Pretty format URLs
\RequirePackage{array}        % More table options
\RequirePackage{multirow}     % Cells spanning multiple rows
\RequirePackage{setspace}     % Change line spacing
\RequirePackage{fancyvrb}     % Pretty code formatting
\RequirePackage{colortbl}     % Color rows in a table
\RequirePackage{afterpage}

\RequirePackage{calc}         % Basic math operations
\RequirePackage{xifthen}      % Conditional code

%-------------------------------------------------------------------------
% Set line spacing
%-------------------------------------------------------------------------

% Increase the line spacing. Normally we don't want to do this because NSF
% allows six lines per vertical inch, but if we want to use up more space
% this is the first thing to change since it makes the proposal easier to
% read.

\linespread{1.0}

%-------------------------------------------------------------------------
% \nsfproposal@setfontsize{pts} Macro
%-------------------------------------------------------------------------
% We explicitly set the font sizes in points so that we know exactly how
% big each font. Subsection headers are supposed to be 11pt, but there
% is no equivalent built-in LaTeX font size which corresponds to 11pt
% (normalsize is 10pt and large is 12pt). Notice we always calculate the
% the line spacing to be 1.2 times the font size.

\newlength{\nsfproposal@Templen}
\newcommand{\nsfproposal@setfontsize}[1]
{%
  \setlength{\nsfproposal@Templen}{#1 * \real{1.2}}%
  \fontsize{#1}{\the\nsfproposal@Templen}\selectfont%
}

\newlength{\cbx@Templen}
\newcommand{\cbxsetfontsize}[1]
{%
  \setlength{\cbx@Templen}{#1 * \real{1.2}}%
  \fontsize{#1}{\the\cbx@Templen}\selectfont%
}

%-------------------------------------------------------------------------
% Caption Fontsizes
%-------------------------------------------------------------------------
% We make the caption font slightly smaller than 11pt to make it
% separate from the text a bit more (smaller caption sizes are allowed
% in the GPG). We also bold the caption label, anticipating that each
% caption will include a short bold title.

\DeclareCaptionFont{slightlysmaller}{\nsfproposal@setfontsize{10pt}}
\captionsetup{font={slightlysmaller},labelfont={bf},skip=8pt}
\captionsetup[sub]{justification=centering,font={slightlysmaller},labelfont={},skip=6pt}
\newcommand{\nsfcaptionsize}{\nsfproposal@setfontsize{10pt}}

% Sometimes we want to manually create subfigure labels. For example we
% might have three subfigures but they are really subplots in a single
% pdf. We can use the following macro to create the subfigure captions.
% It takes two arguments: the first is the width of the subfigure
% caption and the second is the subfigure text. Put your label in the
% subfigure text.

\newcommand{\nsfsubfigcaption}[2]{\subfloat[#2]{\rule{#1}{0pt}}}

%-------------------------------------------------------------------------
% Page layout
%-------------------------------------------------------------------------
% The margins should all be 1in. I wouldn't play with these unless
% absolutely necessary. They probably won't notice 0.05 here and there
% but they might use an automatic margin checker as well.

% If we have annotate turned on then leave extra room in the right margin
% for the margin notes.

\ifthenelse{\equal{\cbxarticle@annotate}{true}}%
{%
  \geometry
  {%
    paperheight=11in,
    paperwidth=9.5in,
    headheight=15pt,
    lmargin=1in,
    rmargin=2in,
    tmargin=1in,
    bmargin=1in,
    marginpar=1.6in,
  }
}{%
  \geometry
  {%
    paperheight=11in,
    paperwidth=8.5in,
    headheight=15pt,
    hmargin=1in,%0.95in,
    tmargin=1in,%0.95in,
    bmargin=1in,%0.95in,
  }
}

%-------------------------------------------------------------------------
% Title blocks
%-------------------------------------------------------------------------
% We have two macros to define the titles for the project summary and
% description parts.

\gdef\nsfproposal@title{}%
\gdef\nsfproposal@titlea{}%
\gdef\nsfproposal@authors{}%

\renewcommand{\title}[1]{\gdef\nsfproposal@title{#1}}
\newcommand{\titlea}[1]{\gdef\nsfproposal@titlea{#1}}
\newcommand{\authors}[1]{\gdef\nsfproposal@authors{#1}}

% Project Description Title

\newcommand{\makedescriptiontitle}
{{%
  \newpage\centering

  {\Large\bf\nsfproposal@title\par}

  \vspace{6pt}
  {\large\nsfproposal@authors\par}

  \thispagestyle{descriptionpagestyle}
  \setcounter{page}{1}
  \vspace{6pt}

}}

%-------------------------------------------------------------------------
% keywords
%-------------------------------------------------------------------------

\newcommand{\keywords}[1]
{%
  \smallskip
  \textbf{\textit{Key Words:}} #1
}

%-------------------------------------------------------------------------
% Section headers
%-------------------------------------------------------------------------
% We change the spacing around the section headers and we tighten up the
% spacing between the number and the title.

\renewcommand{\section}
{%
  \@startsection{section}{1}{\z@}%
                {8pt minus 2pt}%
                {8pt minus 2pt}%
                {\Large\bf}%
}

\renewcommand{\subsection}
{%
  \@startsection{subsection}{2}{\z@}%
                {8pt minus 2pt}%
                {8pt minus 2pt}%
                {\large\bf}%
}

\renewcommand{\subsubsection}
{%
  \@startsection{subsubsection}{3}{\z@}%
                {6pt minus 2pt}%
                {6pt minus 2pt}%
                {\bf}%
}

\renewcommand{\@seccntformat}[1]{\csname the#1\endcsname.\hspace{0.5em}}
\renewcommand{\@seccntformat}[2]{\csname the#1\endcsname.\hspace{0.5em}}

%-------------------------------------------------------------------------
% Float spacings
%-------------------------------------------------------------------------

\renewcommand{\topfraction}{0.95}
\renewcommand{\bottomfraction}{0.95}
\renewcommand{\textfraction}{0.05}
\renewcommand{\floatpagefraction}{0.9}

\newcommand{\bsk}{\baselineskip}
\setlength{\floatsep}{1.2\bsk plus 0.5\bsk minus 0.2\bsk}
\setlength{\textfloatsep}{1\bsk plus 0.5\bsk minus 0.2\bsk}

%-------------------------------------------------------------------------
% Process class options
%-------------------------------------------------------------------------

% Default headers

\renewcommand{\headrulewidth}{0.0in}
\newcommand{\nsfproposal@summaryheader}{}
\newcommand{\nsfproposal@summaryfooter}{}
\newcommand{\nsfproposal@descriptionheader}{}
\newcommand{\nsfproposal@descriptionfooter}{}

% Draft Copy

\ifthenelse{ \equal{\nsfproposal@draftcopy}{true} }
{%
  \RequirePackage[firstpage]{draftwatermark}
  \SetWatermarkLightness{0.9}
  \renewcommand{\nsfproposal@summaryheader}
  {%
    \itshape --- Draft Copy: Please Do Not Redistribute ---%
  }
  \renewcommand{\nsfproposal@descriptionheader}
  {%
    \itshape --- Draft Copy: Please Do Not Redistribute ---%
  }
}{}

% Page Numbers

\ifthenelse{ \equal{\nsfproposal@draftcopy}{true} }
{%
  \renewcommand{\nsfproposal@summaryfooter}{}%
}{%
  \renewcommand{\nsfproposal@summaryfooter}{\thepage}%
}

\renewcommand{\nsfproposal@descriptionfooter}{\thepage}

%-------------------------------------------------------------------------
% Setup page styles
%-------------------------------------------------------------------------

\fancypagestyle{summarypagestyle}
{%
  \fancyhead{}
  \fancyfoot{}
  \fancyhead[C]{\nsfproposal@summaryheader}
  \fancyfoot[C]{\nsfproposal@summaryfooter}
}

\fancypagestyle{descriptionpagestyle}
{%
  \fancyhead{}
  \fancyfoot{}
  \fancyhead[C]{\nsfproposal@descriptionheader}
  \fancyfoot[C]{\nsfproposal@descriptionfooter}
}

\pagestyle{descriptionpagestyle}

%-------------------------------------------------------------------------
% Bibliography
%-------------------------------------------------------------------------

\let\cbxarticle@oldthebibliography\thebibliography
\renewcommand{\thebibliography}[1]
{%
  \ifthenelse{ \equal{\nsfproposal@draftcopy}{true} }
  {%
    \thispagestyle{descriptionpagestyle}
  }{%
    \thispagestyle{descriptionpagestyle}
    %\setcounter{page}{1}
  }
  \cbxarticle@oldthebibliography{#1}%
  \setlength{\itemsep}{0.2ex}%
  \cbxsetfontsize{9.5pt}%
  \vspace{0.7ex}%
}

%-------------------------------------------------------------------------
% Annotate
%-------------------------------------------------------------------------

\newcommand{\annotate}
{%
  \@ifstar\annotateStar\annotateNoStar%
}

\newcommand{\annotateNoStar}[1]
{%
  \ifthenelse{\equal{\cbxarticle@annotate}{true}}%
  {%
    {\color{red}$^\star$}\hspace{-0.02in}%
    \marginpar%
    {%
      \footnotesize%
      \raggedright%
      \begin{cbxlist}[{\large\color{red}$\star$}]{1em}{0em}{0em}%
        \item\color{red}#1%
      \end{cbxlist}%
    }%
  }{}%
}

\newcommand{\annotateStar}[1]
{%
  \ifthenelse{\equal{\cbxarticle@annotate}{true}}%
  {%
    \marginpar%
    {%
      \footnotesize%
      \raggedright%
      \begin{cbxlist}[{\large\color{red}$\star$}]{1em}{0em}{0em}%
        \item\color{red}#1%
      \end{cbxlist}%
    }%
  }%
}

%-------------------------------------------------------------------------
% Code block formatting
%-------------------------------------------------------------------------

\renewcommand{\FancyVerbFormatLine}[1]
{%
  \parbox{1.5ex}{\raggedleft\theFancyVerbLine} #1%
}

%-------------------------------------------------------------------------
% Emph
%-------------------------------------------------------------------------

\definecolor{cbxredC}{RGB}{176,  24,  24}
\newcommand{\highlight}[1]
{%
  \textbf{\color{cbxredC}#1}%
}

%-------------------------------------------------------------------------
% New Paragraph Format
%-------------------------------------------------------------------------

\renewcommand{\paragraph}[1]
{%
  \vspace{0.05in}\textbf{#1} --%
}

%-------------------------------------------------------------------------
% Compact formatting
%-------------------------------------------------------------------------

\newcommand{\TT}[1]{\texttt{#1}}
\newcommand{\BF}[1]{\textbf{#1}}
\newcommand{\IT}[1]{\textit{#1}}
\newcommand{\RM}[1]{\textrm{#1}}

%-------------------------------------------------------------------------
% URL formatting
%-------------------------------------------------------------------------
% Define a new style for url package that will use a smaller font

\makeatletter
\def\url@leostyle{%
  \@ifundefined{selectfont}{\def\UrlFont{\sf}}{\def\UrlFont{\small\ttfamily}}}
\makeatother
\urlstyle{leo}

%-------------------------------------------------------------------------
% Table row formatting
%-------------------------------------------------------------------------

\newcolumntype{+}
{%
  >{\global\let\currentrowstyle\relax}%
}

\newcolumntype{^}
{%
  >{\currentrowstyle}%
}

\newcommand{\rs}
{%
  \gdef\currentrowstyle{\lastrowstyle}%
  \lastrowstyle\ignorespaces%
}

\newcommand{\rowstyle}[1]
{%
  \gdef\lastrowstyle{#1}%
  \gdef\currentrowstyle{#1}%
  #1\ignorespaces%
}

%-------------------------------------------------------------------------
% Useful macros
%-------------------------------------------------------------------------

\newenvironment{cbxlist}[4][\textbullet]
{%
  \begin{list}{#1}
  {%
    \setlength{\leftmargin}{#2}
    \setlength{\rightmargin}{#3}
    \setlength{\topsep}{0pt}
    \setlength{\parsep}{0pt}
    \setlength{\listparindent}{0pt}
    \setlength{\itemsep}{#4}
  }
}{%
  \end{list}
}

\newenvironment{cbxindent}[2]
{%
  \begin{list}{}
  {
    \setlength{\leftmargin}{#1}
    \setlength{\rightmargin}{#2}
    \item
  }
}{%
  \end{list}
}

\newcommand{\fixme}[1]{\textcolor{red}{\small [~#1~]}}
\newcommand{\wu}[2]{\mbox{#1\,#2}}
\newcommand{\um}{\mbox{\textmu m}}
\newcommand{\X}{$\times$}
\newcommand{\x}{\thinspace$\times$\thinspace}
\newcommand{\by}{$\times$}
\newcommand{\texttilde}{{\char'176}}
\newcommand{\tw}{\textwidth}
\newcommand{\cw}{\columnwidth}
\newcommand{\tmidrule}{\midrule[\heavyrulewidth]}

\newcommand{\mmsq}{mm\tsup{2}}
\newcommand{\umsq}{\textmu{}m\tsup{2}}

\newcommand{\tsup}[1]{\textsuperscript{#1}}
\newcommand{\tsub}[1]{\textsubcript{#1}}

\clubpenalty=10000
\widowpenalty=10000

\newcommand{\setdisplayskips}
{%
  \setlength{\abovedisplayskip}{5pt}
  \setlength{\belowdisplayskip}{5pt}
  \setlength{\abovedisplayshortskip}{2pt}
  \setlength{\belowdisplayshortskip}{2pt}
}

