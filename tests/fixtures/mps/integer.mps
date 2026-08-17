* Integer markers.  XCONT is declared before the block and stays continuous,
* XINT and YINT fall inside it, ZCONT comes after INTEND.  WBIN is continuous
* by marker and integer by its BV bound.
NAME          MIXED
ROWS
 N  COST
 L  CAP
COLUMNS
    XCONT     COST             1.0   CAP              1.0
    MARKER                 'MARKER'                 'INTORG'
    XINT      COST             2.0   CAP              1.0
    YINT      COST             3.0   CAP              2.0
    MARKER                 'MARKER'                 'INTEND'
    ZCONT     COST             4.0   CAP              1.0
    WBIN      COST             5.0   CAP              1.0
RHS
    RHS       CAP             10.0
BOUNDS
 UI BND       XINT             8.0
 BV BND       WBIN
ENDATA
