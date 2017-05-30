\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{ferseminar}[2011/04/14 Predlozak za seminar na FER-u]
\LoadClass[a4paper,12pt,pdflatex]{article}
\usepackage[left=3cm,right=2cm,top=3cm,bottom=2cm,nohea dfoot]{geometry}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
%\usepackage[english]{babel}
\usepackage{helvet}
\usepackage{graphicx}
\usepackage{amsmath,amssymb,amsfonts,textcomp}
\usepackage{color}
\usepackage{calc}
\usepackage{url}
\usepackage{hyperref}
\hypersetup{colorlinks=true, linkcolor=black, filecolor=black, pagecolor=black, urlcolor=black, citecolor=black}
\usepackage{indentfirst}
% defaultni font
\renewcommand{\familydefault}{\sfdefault}
% Pages styles (master pages)
% Paket za hrvatski jezik neispravno izostavlja točke iza naslova,
% to ispravljamo zadavanjem novih funkcija za numeriranje koje stavljaju
% točku.
\makeatletter
\renewcommand\thesection{\@arabic\c@section.}
\renewcommand\thesubsection{\thesection\@arabic\c@subsection.}
\renewcommand\thesubsubsection{\thesubsection\@arabic\c@subsubsection.}
\renewcommand\theequation{\@arabic\c@equation}
%\renewcommand\thefigure{\@arabic\c@figure.}
\renewcommand\thetable{\@arabic\c@table.}
\makeatother

% dodavanje točkica u sadržaju
\usepackage{tocloft}
\renewcommand{\cftsecleader}{\cftdotfill{\cftdotsep}}

\usepackage{setspace}
\usepackage{pagedsections} % ovaj paket osigurava da svako novo poglavlje ide na novu stranicu
\usepackage{bibliographyassection} % ovaj paket sredjuje da literatura bude kao section, tj. numerirana i na svojoj stranici

% ovo su definicije varijabli koje treba popuniti
\newcommand{\student}[1]{\def \Student {#1}}
\newcommand{\voditelj}[1]{\def \Voditelj {#1}}
\newcommand{\mjestodatum}[3]{\def \MjestoDatum {#1, #2, #3.}}
\newcommand{\naslov}[1]{\def \Naslov {#1}}

% ovo je naredba za stvaranje prve stranice i stranice sa sadrzajem
\newcommand{\stvoripredstranice}{
  \thispagestyle{empty}

  \vspace{1cm}
  {\centering\mdseries\large
  Zagreb University
  \par}

  \vspace{0.5cm}
  {\centering\bfseries\large
  FACULTY OF ELECTRICAL ENGINEERING AND COMPUTING
  \par}

  \vspace{7.2cm}
  {\centering\bfseries\large
  SEMINAR
  \par}

  \vspace{1cm}
  {\centering\bfseries\Large
  \Naslov
  \par}

  \vspace{0.5cm}
  {\centering\itshape
  \Student
  \par}

  \vspace{0.3cm}
  {\centering\itshape
  \textup{Mentor:} \Voditelj
  \par}


  \vspace{8.5cm}
  {\centering\large
  \MjestoDatum
  \par}


  \pagebreak
  \onehalfspacing % prored 1.5
  % dodavanje sadržaja
  \setcounter{tocdepth}{4} % do 4. razine sadržaja
  \tableofcontents
  \thispagestyle{empty} % ne želimo broj stranice i defaultni format na stranici sa sadržajem

  % uključi datoteke s poglavljima
  \setcounter{page}{0} % uvod pocinje od stranice 1
}

\newcommand{\dodajliteraturu}[1]{
  \nocite{*}
  \bibliographystyle{plain}
  \bibliography{#1}
}
