<?

switch($type_of_search)
{
	case 'doc':
		header("Location:http://google.com/search?q=site%3Afann.sourceforge.net+" . rawurlencode(stripslashes($words)));
		break;
	case 'mlists':
		header("Location:http://sourceforge.net/search/?type_of_search=$type_of_search&group_id=93562&forum_id=37468&words=" . rawurlencode(stripslashes($words)));
		break;
	case 'forums':
		header("Location:http://sourceforge.net/search/?type_of_search=$type_of_search&group_id=93562&forum_id=323465&words=" . rawurlencode(stripslashes($words)));
		break;
}

?>